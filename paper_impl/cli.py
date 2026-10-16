from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
from .features import FEATURE_NAMES, extract_features
from .metrics import mask_metrics, physical_area_cm2
from .model import load_model, predict_image, train_model
from .segmentation import PaperSegmenter, SegmentationConfig, read_image


def add_segmentation_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--entropy-window", type=int, default=9)
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed normalized threshold [0,1]; default auto-searches candidates",
    )
    parser.add_argument(
        "--polarity",
        choices=("auto", "normal", "inverted"),
        default="auto",
        help="Threshold direction; auto also searches entropy percentiles",
    )
    parser.add_argument("--min-object-area", type=int, default=1500)
    parser.add_argument("--close-kernel", type=int, default=9)
    parser.add_argument("--close-iterations", type=int, default=1)
    parser.add_argument("--keep-all", action="store_true", help="Do not retain only the maximum domain")
    parser.add_argument("--nondefect-ratio", type=float, default=0.03)


def config_from_args(args: argparse.Namespace) -> SegmentationConfig:
    return SegmentationConfig(
        entropy_window=args.entropy_window,
        threshold=args.threshold,
        polarity=args.polarity,
        min_object_area=args.min_object_area,
        close_kernel=args.close_kernel,
        close_iterations=args.close_iterations,
        keep_largest=not args.keep_all,
        nondefect_ratio=args.nondefect_ratio,
    )


def write_stages(output: Path, image: np.ndarray, result: object) -> None:
    output.mkdir(parents=True, exist_ok=True)
    stages = {
        "01_original.png": image,
        "02_texture_entropy.png": result.entropy_gray,
        "03_binary.png": result.binary,
        "04_remove_small_objects.png": result.area_filtered,
        "05_morphological_close.png": result.morph_closed,
        "06_filled_components.png": result.all_components,
        "07_maximum_domain.png": result.mask,
        "08_overlay.png": result.overlay,
    }
    for name, stage in stages.items():
        if not cv2.imwrite(str(output / name), stage):
            raise OSError(f"Could not write {output / name}")


def command_segment(args: argparse.Namespace) -> None:
    image = read_image(args.image)
    result = PaperSegmenter(config_from_args(args)).segment(image)
    features = extract_features(result.all_components, result.mask)
    write_stages(Path(args.output), image, result)
    report = {
        "is_defect": result.is_defect,
        "foreground_ratio": result.foreground_ratio,
        "selected_threshold": result.threshold_value,
        "selected_polarity": result.polarity,
        "features": dict(zip(FEATURE_NAMES, map(float, features))),
    }
    if (args.scale_length_cm is None) != (args.scale_length_pixels is None):
        raise ValueError("Provide both --scale-length-cm and --scale-length-pixels")
    if args.scale_length_cm is not None:
        report["physical_area_cm2"] = physical_area_cm2(
            features[3], args.scale_length_cm, args.scale_length_pixels
        )
    print(json.dumps(report, indent=2))


def command_train(args: argparse.Namespace) -> None:
    metrics = train_model(args.dataset, args.model, config_from_args(args), args.seed)
    print(json.dumps(metrics, indent=2))


def command_predict(args: argparse.Namespace) -> None:
    image = read_image(args.image)
    artifact = load_model(args.model)
    label, name, features, result = predict_image(image, artifact)
    if args.output:
        write_stages(Path(args.output), image, result)
    print(json.dumps({"label": label, "class": name, "features": dict(zip(FEATURE_NAMES, map(float, features)))}, indent=2))


def command_evaluate_mask(args: argparse.Namespace) -> None:
    predicted = cv2.imread(args.prediction, cv2.IMREAD_GRAYSCALE)
    reference = cv2.imread(args.reference, cv2.IMREAD_GRAYSCALE)
    if predicted is None or reference is None:
        raise ValueError("Could not read prediction or reference mask")
    if predicted.shape != reference.shape:
        raise ValueError("Prediction and reference masks must have the same dimensions")
    print(json.dumps(mask_metrics(predicted, reference), indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Gao et al. 2020 pothole method reproduction")
    subparsers = parser.add_subparsers(dest="command", required=True)

    segment = subparsers.add_parser("segment", help="Run the paper segmentation stages")
    segment.add_argument("image")
    segment.add_argument("--output", default="paper_output")
    segment.add_argument("--scale-length-cm", type=float)
    segment.add_argument("--scale-length-pixels", type=float)
    add_segmentation_arguments(segment)
    segment.set_defaults(function=command_segment)

    train = subparsers.add_parser("train", help="Train the four-class LIBSVM model")
    train.add_argument("dataset")
    train.add_argument("--model", default="paper_model.joblib")
    train.add_argument("--seed", type=int, default=42)
    add_segmentation_arguments(train)
    train.set_defaults(function=command_train)

    predict = subparsers.add_parser("predict", help="Classify an image with a trained model")
    predict.add_argument("image")
    predict.add_argument("--model", default="paper_model.joblib")
    predict.add_argument("--output")
    predict.set_defaults(function=command_predict)

    evaluate = subparsers.add_parser("evaluate-mask", help="Compare a predicted mask to ground truth")
    evaluate.add_argument("prediction")
    evaluate.add_argument("reference")
    evaluate.set_defaults(function=command_evaluate_mask)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()

