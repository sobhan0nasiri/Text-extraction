import argparse
import glob
import logging
import multiprocessing as mp
import os
import ssl
import sys
import time
import cv2
import numpy as np
import torch

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

from pipeline import OCRPipeline
from parallel_utils import get_shared_executor
from strategies.corner_detection import DocAligner_RealWeightsDetector
from strategies.obstacle import RTDETR_RealWeightsDetector
from strategies.rectification import DynamicRectifier
from strategies.input_source import FileInputSource, CameraFrameInputSource
from strategies.text_detection import (
    DocTR_RealWeightsDetector,
    DocTR_MultiArchDetector,
    CRAFT_TextDetector,
    EAST_TextDetector,
    MultiModelTextDetector,
)


def _decode_image_worker(path):
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        return path, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return path, np.ascontiguousarray(img_rgb)


def _write_result(out_path, results):
    with open(out_path, "w", encoding="utf-8") as f:
        if results:
            for r in results:
                f.write(r["text"] + "\n")
        else:
            f.write("")


def _build_recognizer_replicas(build_one, num_replicas):
    num_replicas = max(1, num_replicas)
    replicas = [build_one() for _ in range(num_replicas)]
    if len(replicas) == 1:
        return replicas[0]
    from strategies.text_recognition_parallel import ParallelTextRecognizer
    return ParallelTextRecognizer(replicas)


def build_fallback_recognizer(args):
    if args.fallback_recognizer == "none":
        return None
    elif args.fallback_recognizer == "trocr-small":
        from strategies.text_recognition import TrOCR_RealWeightsRecognizer
        batch_size = args.recognizer_batch_size or 16

        def _build():
            return TrOCR_RealWeightsRecognizer(
                batch_size=batch_size, num_beams=1,
                model_name="microsoft/trocr-small-printed",
                use_fp16=not args.no_fp16,
            )
        return _build_recognizer_replicas(_build, args.recognizer_replicas)
    elif args.fallback_recognizer == "ppocrv5":
        from strategies.text_recognition_ppocr import PPOCRv5_Recognizer
        return PPOCRv5_Recognizer(server_urls=args.ppocr_server_urls)
    else:
        from strategies.text_recognition_fast import DocTR_FastRecognizer
        batch_size = args.recognizer_batch_size or 32

        def _build():
            return DocTR_FastRecognizer(arch=args.fallback_recognizer, use_fp16=not args.no_fp16,
                                        batch_size=batch_size)
        return _build_recognizer_replicas(_build, args.recognizer_replicas)


def build_recognizer(args):
    if args.recognizer == "trocr":
        from strategies.text_recognition import TrOCR_RealWeightsRecognizer
        batch_size = args.recognizer_batch_size or 32

        def _build():
            return TrOCR_RealWeightsRecognizer(
                batch_size=batch_size,
                num_beams=args.beams,
                model_name=f"microsoft/trocr-{args.trocr_size}-printed",
                use_fp16=not args.no_fp16,
            )
        primary = _build_recognizer_replicas(_build, args.recognizer_replicas)
        return primary, build_fallback_recognizer(args)
    elif args.recognizer == "fast":
        from strategies.text_recognition_fast import DocTR_FastRecognizer
        batch_size = args.recognizer_batch_size or 32

        def _build():
            return DocTR_FastRecognizer(arch=args.fast_arch, use_fp16=not args.no_fp16,
                                        batch_size=batch_size)
        primary = _build_recognizer_replicas(_build, args.recognizer_replicas)
        return primary, build_fallback_recognizer(args)
    elif args.recognizer == "ppocrv5":
        from strategies.text_recognition_ppocr import PPOCRv5_Recognizer
        primary = PPOCRv5_Recognizer(server_urls=args.ppocr_server_urls)
        return primary, build_fallback_recognizer(args)
    else:
        raise ValueError(f"Unknown recognizer '{args.recognizer}'")

def build_text_detector(args):
    if args.detector == "dbnet":
        return DocTR_RealWeightsDetector(use_fp16=not args.no_fp16)
    elif args.detector == "ensemble":
        return DocTR_MultiArchDetector(use_fp16=not args.no_fp16)
    elif args.detector == "craft":
        return CRAFT_TextDetector()
    elif args.detector == "east":
        return EAST_TextDetector(model_path=args.east_model)
    elif args.detector == "full":
        return MultiModelTextDetector([
            DocTR_MultiArchDetector(use_fp16=not args.no_fp16),
            CRAFT_TextDetector(),
            EAST_TextDetector(model_path=args.east_model),
        ])
    raise ValueError(f"Unknown detector '{args.detector}'")

def main():
    parser = argparse.ArgumentParser(description="OCR Pipeline")
    parser.add_argument("--mode", type=str, choices=["camera", "file", "folder"], default="camera")
    parser.add_argument("--file", type=str, help="Path to image file (required if mode='file')")
    parser.add_argument("--input-dir", type=str, help="Directory of images to process (required if mode='folder')")
    parser.add_argument("--output-dir", type=str, default="ocr_results", help="Where per-file .txt results are written in folder mode")
    parser.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1),
                        help="CPU worker processes used to decode/prefetch images in folder mode, overlapped with GPU inference")

    parser.add_argument("--obstacle-backbone", type=str, default="r18vd",
                        choices=["r18vd", "r34vd", "r50vd", "r101vd"],
                        help="RT-DETRv2 backbone for the screen/obstacle detector.")

    parser.add_argument("--recognizer", type=str, default="fast", choices=["trocr", "fast", "ppocrv5"],
                        help="'trocr' = optimized TrOCR. 'fast' = docTR-based recognizer. "
                            "'ppocrv5' = PaddleOCR, served by an isolated microservice.")
    parser.add_argument("--trocr-size", type=str, default="base", choices=["small", "base", "large"])
    parser.add_argument("--beams", type=int, default=1)
    parser.add_argument("--fast-arch", type=str, default="parseq",
                        choices=["parseq", "master", "crnn_mobilenet_v3_small", "crnn_vgg16_bn", "crnn_mobilenet_v3_large", "vitstr_small"])
    parser.add_argument("--no-fp16", action="store_true")
    parser.add_argument("--realtime-budget", type=float, default=2.5)
    parser.add_argument("--adaptive-resolution", action="store_true",
                        help="Allow the pipeline to auto-adjust detector inference resolution based on the realtime budget. Off by default for reproducible timing/results.")
    parser.add_argument("--ppocr-server-urls", type=str, default="http://127.0.0.1:5005",
                        help="Comma-separated base URLs of one or more PP-OCRv5 microservice ports "
                            "(only used when --recognizer/--fallback-recognizer ppocrv5). For a single image, "
                            "the detected word boxes are split evenly across these ports and sent as concurrent "
                            "HTTP requests, e.g. 'PPOCR_PORT=5005 python ppocr_server.py' and "
                            "'PPOCR_PORT=5006 python ppocr_server.py', then "
                            "'--ppocr-server-urls http://127.0.0.1:5005,http://127.0.0.1:5006'. Each port can "
                            "ALSO host several model replicas internally on its own (single-port, multi-model): "
                            "set PPOCR_NUM_REPLICAS=N (and optionally PPOCR_DEVICES=gpu:0,gpu:1,... to spread "
                            "replicas across GPUs, and PPOCR_REPLICA_BATCH_SIZE to cap each replica's batch) "
                            "before starting that ppocr_server.py process. Total parallel PP-OCR workers = "
                            "(number of ports) x (PPOCR_NUM_REPLICAS per port).")

    parser.add_argument("--recognizer-replicas", type=int, default=1,
                        help="Number of in-process model replicas to run concurrently on the GPU for the "
                            "text-recognition stage of a SINGLE image: the detected word/line boxes are split "
                            "evenly across these replicas and processed in parallel via CUDA streams, then "
                            "merged back in order. Applies to --recognizer/--fallback-recognizer 'trocr', "
                            "'trocr-small' and the docTR 'fast' architectures. Each replica loads its own full "
                            "copy of the model weights, so GPU memory usage scales roughly linearly with this "
                            "value. Has no effect on 'ppocrv5', whose parallelism is controlled instead by "
                            "--ppocr-server-urls (process/port-level) and PPOCR_NUM_REPLICAS on the server side "
                            "(in-process/thread-level).")
    parser.add_argument("--recognizer-batch-size", type=int, default=None,
                        help="Per-replica batch size for text recognition (default: 32 for 'trocr'/'fast', "
                            "16 for 'trocr-small'). When --recognizer-replicas > 1, each replica processes its "
                            "own share of boxes in batches of this size.")

    parser.add_argument("--fallback-recognizer", type=str, default="none",
                        choices=["none", "trocr-small", "ppocrv5", "parseq", "master", "crnn_mobilenet_v3_small", "crnn_vgg16_bn", "crnn_mobilenet_v3_large", "vitstr_small"],
                        help="Second-opinion recognizer applied only to low-confidence/short words from --recognizer fast.")

    parser.add_argument("--detector", type=str, default="dbnet",
                        choices=["dbnet", "ensemble", "craft", "east", "full"],
                        help="'dbnet' = single docTR DBNet. 'ensemble' = multi-arch docTR. "
                            "'craft' / 'east' = standalone. 'full' = all detectors combined.")
    parser.add_argument("--east-model", type=str, default="frozen_east_text_detection.pb")
    parser.add_argument("--debug", action="store_true",
                        help="Save all stepN debug images to disk.")
    parser.add_argument("--log-level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    parser.add_argument("--quiet", action="store_true", help="Disable all log output.")

    args = parser.parse_args()

    torch.backends.cudnn.benchmark = (args.mode in ("camera", "folder"))

    logging.basicConfig(level=getattr(logging, args.log_level),
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    if args.quiet:
        logging.disable(logging.CRITICAL)
    logger = logging.getLogger(__name__)

    logger.info("Initializing models. This may take a while to download weights on the first run...")
    corner_detector = DocAligner_RealWeightsDetector(debug=args.debug)
    obstacle_detector = RTDETR_RealWeightsDetector(backbone=args.obstacle_backbone, use_fp16=not args.no_fp16)
    rectifier = DynamicRectifier()
    text_detector = build_text_detector(args)
    result = build_recognizer(args)
    text_recognizer, fallback_recognizer = result if isinstance(result, tuple) else (result, None)

    pipeline = OCRPipeline(
    corner_detector, obstacle_detector, rectifier, text_detector, text_recognizer,
    fallback_recognizer=fallback_recognizer,
    realtime_budget_seconds=args.realtime_budget, 
    debug=args.debug,
    enable_adaptive_resolution=args.adaptive_resolution,
    crop_pad_ratio=0.55
)

    if args.mode == "file":
        if not args.file:
            logger.error("--file argument is required when mode='file'")
            sys.exit(1)
        try:
            file_input = FileInputSource(file_path=args.file)
            image_tensor = file_input.get_frame()

            if image_tensor is not None:

                results = pipeline.process_image(image_tensor)

                if results:
                    print("\n[RESULT] Extracted Texts (Line by Line):")
                    for r in results:
                        print(f"Line {r['line_id']} ({r['words_count']} words): {r['text']}")

                else:
                    print("[RESULT] No text detected.")
        except Exception:
            logger.exception("Processing failed")

    elif args.mode == "folder":
        if not args.input_dir:
            logger.error("--input-dir argument is required when mode='folder'")
            sys.exit(1)

        exts = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff")
        paths = sorted(p for e in exts for p in glob.glob(os.path.join(args.input_dir, e)))
        if not paths:
            logger.error(f"No images found in {args.input_dir}")
            sys.exit(1)

        os.makedirs(args.output_dir, exist_ok=True)
        logger.info(f"Folder mode: {len(paths)} images, {args.workers} CPU decode workers overlapped with GPU inference.")

        ctx = mp.get_context("spawn")
        t_start = time.time()
        processed = 0
        io_executor = get_shared_executor()
        io_futures = []

        with ctx.Pool(processes=args.workers) as pool:
            for path, img_rgb in pool.imap(_decode_image_worker, paths, chunksize=1):
                if img_rgb is None:
                    logger.warning(f"Skipping unreadable file: {path}")
                    continue

                image_tensor = torch.from_numpy(img_rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

                try:
                    results = pipeline.process_image(image_tensor)
                except Exception:
                    logger.exception(f"Processing failed for {path}")
                    continue

                stem = os.path.splitext(os.path.basename(path))[0]
                out_path = os.path.join(args.output_dir, f"{stem}.txt")
                io_futures.append(io_executor.submit(_write_result, out_path, results))

                processed += 1
                logger.info(f"[{processed}/{len(paths)}] {path} -> {out_path}")

        for fut in io_futures:
            fut.result()

        elapsed = time.time() - t_start
        logger.info(f"Folder mode done: {processed}/{len(paths)} images in {elapsed:.1f}s "
                    f"({processed / elapsed if elapsed > 0 else 0:.2f} img/s).")

    elif args.mode == "camera":
        camera_input = CameraFrameInputSource(camera_index=0)
        cv2.namedWindow("Live Camera Stream (Press 'q' to stop)", cv2.WINDOW_NORMAL)

        logger.info("Starting Real-time Camera Processing Loop")
        logger.info("Press 's' to SCAN the current frame.")
        logger.info("Press 'q' on the camera window to EXIT.")

        prev_t = time.time()
        last_debug_print = prev_t

        try:
            while True:
                frame_tensor = camera_input.get_frame()
                if frame_tensor is None:
                    logger.debug("frame_tensor is None -> loop exiting")
                    break

                now = time.time()
                fps = 1.0 / (now - prev_t) if now != prev_t else 0
                prev_t = now

                if now - last_debug_print >= 2.0:
                    logger.debug(f"loop fps: {fps:.1f}")
                    last_debug_print = now

                key = cv2.waitKey(15) & 0xFF
                if key != 255:
                    logger.debug(f"key pressed: {key}")

                if key == ord('q'):
                    logger.info("User stopped the camera loop.")
                    break

                elif key == ord('s'):
                    logger.info("'s' pressed! Processing the captured frame...")
                    try:
                        results = pipeline.process_image(frame_tensor)
                        if results:
                            print("\n[RESULT] Extracted Texts (Line by Line):")
                            for r in results:
                                print(f" - {r['text']}")
                            logger.info("Ready for next scan. Press 's' again.")
                        else:
                            print("[RESULT] No text detected.")
                    except Exception as e:
                        logger.error(f"Processing failed: {e}")
        finally:
            camera_input.release()

if __name__ == "__main__":
    main()