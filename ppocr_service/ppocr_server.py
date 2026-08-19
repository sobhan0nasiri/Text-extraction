import os
import struct
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from flask import Flask, jsonify, request
from paddleocr import TextRecognition

app = Flask(__name__)

_RAW_HEADER = struct.Struct(">3I")


def _decode_raw_crop(file_storage):
    data = file_storage.read()
    if len(data) < _RAW_HEADER.size:
        return None
    h, w, c = _RAW_HEADER.unpack_from(data, 0)
    payload = data[_RAW_HEADER.size:]
    if h <= 0 or w <= 0 or c <= 0 or len(payload) != h * w * c:
        return None
    return np.frombuffer(payload, dtype=np.uint8).reshape(h, w, c)


DEVICE = os.environ.get("PPOCR_DEVICE") or ("cpu" if os.environ.get("PPOCR_USE_CPU", "0") == "1" else "gpu:0")
MODEL_NAME = os.environ.get("PPOCR_REC_MODEL", "PP-OCRv5_server_rec")
PORT = int(os.environ.get("PPOCR_PORT", "5005"))
THREADS = int(os.environ.get("PPOCR_THREADS", "4"))

NUM_REPLICAS = max(1, int(os.environ.get("PPOCR_NUM_REPLICAS", "1")))
REPLICA_BATCH_SIZE = max(1, int(os.environ.get("PPOCR_REPLICA_BATCH_SIZE", "64")))

_raw_devices = os.environ.get("PPOCR_DEVICES")
if _raw_devices:
    REPLICA_DEVICES = [d.strip() for d in _raw_devices.split(",") if d.strip()]
    if len(REPLICA_DEVICES) < NUM_REPLICAS:
        REPLICA_DEVICES += [REPLICA_DEVICES[-1]] * (NUM_REPLICAS - len(REPLICA_DEVICES))
else:
    REPLICA_DEVICES = [DEVICE] * NUM_REPLICAS
REPLICA_DEVICES = REPLICA_DEVICES[:NUM_REPLICAS]

if NUM_REPLICAS > 1 and len(set(REPLICA_DEVICES)) < NUM_REPLICAS:
    print(f"[PPOCR-SERVICE][WARNING] PPOCR_NUM_REPLICAS={NUM_REPLICAS} but the resolved devices "
            f"{REPLICA_DEVICES} are not all distinct -- replicas sharing one physical GPU will NOT "
            f"run with real hardware parallelism (they contend for the same SMs/VRAM bandwidth). "
            f"Prefer PPOCR_NUM_REPLICAS=1 with a larger PPOCR_REPLICA_BATCH_SIZE / --recognizer-batch-size "
            f"on a single GPU, or set PPOCR_DEVICES to one distinct device per replica on a multi-GPU box.")

ENABLE_HPI = os.environ.get("PPOCR_ENABLE_HPI", "1") == "1"
USE_TENSORRT = os.environ.get("PPOCR_USE_TENSORRT", "0") == "1"
PRECISION = os.environ.get("PPOCR_PRECISION", "fp16")


def _build_recognizer(device):
    kwargs = {"model_name": MODEL_NAME, "device": device}
    if ENABLE_HPI:
        kwargs["enable_hpi"] = True
    if USE_TENSORRT:
        kwargs["use_tensorrt"] = True
        kwargs["precision"] = PRECISION
    if not (ENABLE_HPI or USE_TENSORRT):
        return TextRecognition(**kwargs)
    try:
        model = TextRecognition(**kwargs)
        print(f"[PPOCR-SERVICE] {MODEL_NAME}@{device} loaded with enable_hpi={ENABLE_HPI}, "
                f"use_tensorrt={USE_TENSORRT}, precision={PRECISION}.")
        return model
    except Exception as e:
        print(f"[PPOCR-SERVICE][WARNING] High-performance inference setup failed for {device} "
                f"({e}); falling back to normal (non-accelerated) inference. Install/verify a "
                f"matching TensorRT + paddlepaddle-gpu build to get the speedup (see PaddleOCR's "
                f"high-performance inference docs). Set PPOCR_ENABLE_HPI=0 to silence this.")
        return TextRecognition(model_name=MODEL_NAME, device=device)


def _warmup_replica(model, batch_size):
    dummy = np.zeros((48, 320, 3), dtype=np.uint8)
    for wb in sorted({1, min(8, batch_size), batch_size}):
        try:
            t0 = time.perf_counter()
            list(model.predict(input=[dummy] * wb, batch_size=wb))
            print(f"[PPOCR-SERVICE] Warm-up predict(batch={wb}) finished in "
                    f"{time.perf_counter() - t0:.2f}s.")
        except Exception as e:
            print(f"[PPOCR-SERVICE][WARNING] Warm-up predict(batch={wb}) failed "
                    f"(server will still work, first real request may be slower): {e}")


print(f"[PPOCR-SERVICE] Loading {NUM_REPLICAS} replica(s) of {MODEL_NAME} on devices {REPLICA_DEVICES}...")
model_pool = [_build_recognizer(REPLICA_DEVICES[i]) for i in range(NUM_REPLICAS)]
pool_executor = ThreadPoolExecutor(max_workers=NUM_REPLICAS)

for _model in model_pool:
    _warmup_replica(_model, REPLICA_BATCH_SIZE)

print(f"[PPOCR-SERVICE] {NUM_REPLICAS} model replica(s) loaded and warmed up "
        f"(per-replica batch size={REPLICA_BATCH_SIZE}). Ready to serve requests on /recognize_batch. "
        f"Tip: start this server with PPOCR_REPLICA_BATCH_SIZE set to whatever you pass as "
        f"--recognizer-batch-size on the client, so warm-up covers the batch size you'll actually use.")


def _extract(res):
    try:
        return res["rec_text"], float(res["rec_score"])
    except (TypeError, KeyError):
        return getattr(res, "rec_text", ""), float(getattr(res, "rec_score", 0.0))


def _split_indices(n_items, n_chunks):
    base, remainder = divmod(n_items, n_chunks)
    chunks, start = [], 0
    for i in range(n_chunks):
        size = base + (1 if i < remainder else 0)
        chunks.append(list(range(start, start + size)))
        start += size
    return chunks


def _run_replica(model, imgs, batch_size):
    if not imgs:
        return [], 0.0
    t0 = time.perf_counter()
    outputs = []
    for start in range(0, len(imgs), batch_size):
        sub_batch = imgs[start:start + batch_size]
        outputs.extend(model.predict(input=sub_batch, batch_size=len(sub_batch)))
    elapsed = time.perf_counter() - t0
    return outputs, elapsed


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_name": MODEL_NAME,
        "device": DEVICE,
        "num_replicas": NUM_REPLICAS,
        "replica_devices": REPLICA_DEVICES,
        "replica_batch_size": REPLICA_BATCH_SIZE,
        "enable_hpi": ENABLE_HPI,
        "use_tensorrt": USE_TENSORRT,
        "precision": PRECISION,
    })


@app.route("/recognize_batch", methods=["POST"])
def recognize_batch():
    t_req0 = time.perf_counter()

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images provided under 'images' field"}), 400

    imgs = []
    for f in files:
        try:
            img_bgr = _decode_raw_crop(f)
            if img_bgr is None:
                raise ValueError("corrupt or empty raw crop payload")
            imgs.append(img_bgr)
        except Exception as e:
            print(f"[PPOCR-SERVICE][WARNING] failed to decode one crop: {e}")
            imgs.append(None)

    req_batch_size = request.form.get("batch_size", type=int)
    effective_batch_size = req_batch_size if req_batch_size and req_batch_size > 0 else REPLICA_BATCH_SIZE

    results = [{"text": "", "conf": 0.0} for _ in imgs]
    valid_indices = [i for i, im in enumerate(imgs) if im is not None]
    valid_imgs = [imgs[i] for i in valid_indices]

    model_time = 0.0
    if valid_imgs:
        index_groups = _split_indices(len(valid_imgs), NUM_REPLICAS)

        futures = {}
        for replica_id, idx_group in enumerate(index_groups):
            if not idx_group:
                continue
            replica_imgs = [valid_imgs[i] for i in idx_group]
            fut = pool_executor.submit(_run_replica, model_pool[replica_id], replica_imgs, effective_batch_size)
            futures[fut] = idx_group

        replica_times = []
        for fut in as_completed(futures):
            idx_group = futures[fut]
            try:
                outputs, elapsed = fut.result()
            except Exception as e:
                print(f"[PPOCR-SERVICE][WARNING] replica batch failed ({len(idx_group)} crops): {e}")
                continue

            replica_times.append(elapsed)
            for local_i, out_i in enumerate(idx_group):
                if local_i >= len(outputs):
                    break
                text, score = _extract(outputs[local_i])
                results[valid_indices[out_i]] = {"text": text or "", "conf": score}

        model_time = max(replica_times) if replica_times else 0.0

    decode_time = max(0.0, (time.perf_counter() - t_req0) - model_time)

    return jsonify({"results": results, "model_time": model_time, "decode_time": decode_time})


if __name__ == "__main__":
    from waitress import serve
    print(f"[PPOCR-SERVICE] Serving on port {PORT} with {THREADS} waitress threads, "
            f"{NUM_REPLICAS} internal model replica(s) on {REPLICA_DEVICES}.")
    serve(app, host="127.0.0.1", port=PORT, threads=THREADS)