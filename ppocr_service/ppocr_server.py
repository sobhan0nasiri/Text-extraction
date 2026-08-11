import os
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image
from paddleocr import TextRecognition

app = Flask(__name__)

DEVICE = "cpu" if os.environ.get("PPOCR_USE_CPU", "0") == "1" else "gpu:0"
MODEL_NAME = os.environ.get("PPOCR_REC_MODEL", "PP-OCRv5_server_rec")

print(f"[PPOCR-SERVICE] Loading {MODEL_NAME} recognizer (device={DEVICE})...")
model = TextRecognition(model_name=MODEL_NAME, device=DEVICE)
print("[PPOCR-SERVICE] Model loaded. Ready to serve requests on /recognize_batch.")


def _extract(res):
    try:
        return res["rec_text"], float(res["rec_score"])
    except (TypeError, KeyError):
        return getattr(res, "rec_text", ""), float(getattr(res, "rec_score", 0.0))


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_name": MODEL_NAME, "device": DEVICE})


@app.route("/recognize_batch", methods=["POST"])
def recognize_batch():
    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "no images provided under 'images' field"}), 400

    imgs = []
    for f in files:
        try:
            pil_img = Image.open(f.stream).convert("RGB")
            img_bgr = np.array(pil_img)[:, :, ::-1]
            imgs.append(img_bgr)
        except Exception as e:
            print(f"[PPOCR-SERVICE][WARNING] failed to decode one crop: {e}")
            imgs.append(None)

    results = [{"text": "", "conf": 0.0} for _ in imgs]
    valid_indices = [i for i, im in enumerate(imgs) if im is not None]
    valid_imgs = [imgs[i] for i in valid_indices]

    model_time = 0.0
    if valid_imgs:
        try:
            t0 = time.perf_counter()
            outputs = model.predict(input=valid_imgs, batch_size=len(valid_imgs))
            model_time = time.perf_counter() - t0
            for idx, res in zip(valid_indices, outputs):
                text, score = _extract(res)
                results[idx] = {"text": text or "", "conf": score}
        except Exception as e:
            print(f"[PPOCR-SERVICE][WARNING] batch recognition failed ({len(valid_imgs)} crops): {e}")

    return jsonify({"results": results, "model_time": model_time})


if __name__ == "__main__":
    from waitress import serve
    serve(app, host="127.0.0.1", port=5005, threads=1)