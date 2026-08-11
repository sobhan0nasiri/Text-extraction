import argparse
import logging
import ssl
import sys
import time
import cv2
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


def build_fallback_recognizer(args):
    if args.fallback_recognizer == "none":
        return None
    elif args.fallback_recognizer == "trocr-small":
        from strategies.text_recognition import TrOCR_RealWeightsRecognizer
        return TrOCR_RealWeightsRecognizer(
            batch_size=16, num_beams=1,
            model_name="microsoft/trocr-small-printed",
            use_fp16=not args.no_fp16,
        )
    elif args.fallback_recognizer == "ppocrv5":
        from strategies.text_recognition_ppocr import PPOCRv5_Recognizer
        return PPOCRv5_Recognizer(server_url=args.ppocr_server_url)
    else:
        from strategies.text_recognition_fast import DocTR_FastRecognizer
        return DocTR_FastRecognizer(arch=args.fallback_recognizer, use_fp16=not args.no_fp16)


def build_recognizer(args):
    if args.recognizer == "trocr":
        from strategies.text_recognition import TrOCR_RealWeightsRecognizer
        primary = TrOCR_RealWeightsRecognizer(
            batch_size=32,
            num_beams=args.beams,
            model_name=f"microsoft/trocr-{args.trocr_size}-printed",
            use_fp16=not args.no_fp16,
        )
        return primary, build_fallback_recognizer(args)
    elif args.recognizer == "fast":
        from strategies.text_recognition_fast import DocTR_FastRecognizer
        primary = DocTR_FastRecognizer(arch=args.fast_arch, use_fp16=not args.no_fp16)
        return primary, build_fallback_recognizer(args)
    elif args.recognizer == "ppocrv5":
        from strategies.text_recognition_ppocr import PPOCRv5_Recognizer
        primary = PPOCRv5_Recognizer(server_url=args.ppocr_server_url)
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
    parser.add_argument("--mode", type=str, choices=["camera", "file"], default="camera")
    parser.add_argument("--file", type=str, help="Path to image file (required if mode='file')")

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
    parser.add_argument("--ppocr-server-url", type=str, default="http://127.0.0.1:5005",
                        help="Base URL of the isolated PP-OCRv5 microservice "
                            "(only used when --recognizer ppocrv5). Start it with "
                            "'python ppocr_service/ppocr_server.py' inside its own venv.")

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

    torch.backends.cudnn.benchmark = (args.mode == "camera")
    
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