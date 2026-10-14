"""LIBSVM-compatible multiclass training and evaluation."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import cv2
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

from .features import FEATURE_NAMES, extract_features
from .segmentation import PaperSegmenter, SegmentationConfig


CLASS_NAMES = {1: "pothole", 2: "transverse_crack", 3: "longitudinal_crack", 4: "complex_crack"}
NAME_TO_CLASS = {name: label for label, name in CLASS_NAMES.items()}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def iter_dataset(root: str | Path) -> Iterable[tuple[Path, int]]:
    root = Path(root)
    for label, name in CLASS_NAMES.items():
        folder = root / name
        if not folder.is_dir():
            raise ValueError(f"Missing class directory: {folder}")
        for path in sorted(folder.rglob("*")):
            if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
                yield path, label


def build_feature_dataset(root: str | Path, segmenter: PaperSegmenter) -> tuple[np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    paths: list[str] = []
    for path, label in iter_dataset(root):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Could not read image: {path}")
        result = segmenter.segment(image)
        features.append(extract_features(result.all_components, result.mask))
        labels.append(label)
        paths.append(str(path))
    if not features:
        raise ValueError(f"No images found under {root}")
    return np.vstack(features), np.asarray(labels, dtype=np.int32), paths


def train_model(
    dataset_root: str | Path,
    model_path: str | Path,
    config: SegmentationConfig | None = None,
    seed: int = 42,
) -> dict[str, object]:
    """Train the paper's 4-class, 70/30 C-SVC and persist it with metadata."""
    segmenter = PaperSegmenter(config)
    x, y, paths = build_feature_dataset(dataset_root, segmenter)
    if len(np.unique(y)) != 4:
        raise ValueError("Training requires all four paper classes")
    if any(np.count_nonzero(y == label) < 2 for label in CLASS_NAMES):
        raise ValueError("Each class needs at least two images for a stratified 70/30 split")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.30, random_state=seed, stratify=y
    )
    # sklearn.svm.SVC is backed by LIBSVM. These are LIBSVM C-SVC defaults:
    # RBF kernel, C=1, gamma=1/number_of_features, one-vs-one multiclass.
    classifier = SVC(C=1.0, kernel="rbf", gamma=1.0 / len(FEATURE_NAMES), decision_function_shape="ovo")
    classifier.fit(x_train, y_train)
    prediction = classifier.predict(x_test)

    # Equation (19) treats pothole as positive and all crack classes as negative.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test == 1,
        prediction == 1,
        average="binary",
        pos_label=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": float(accuracy_score(y_test, prediction)),
        "pothole_precision": float(precision),
        "pothole_recall": float(recall),
        "pothole_f1": float(f1),
        "train_samples": int(len(y_train)),
        "validation_samples": int(len(y_test)),
    }
    artifact = {
        "classifier": classifier,
        "segmentation_config": asdict(segmenter.config),
        "feature_names": FEATURE_NAMES,
        "class_names": CLASS_NAMES,
        "metrics": metrics,
        "source_paths": paths,
        "paper": "Gao et al. (2020), doi:10.1155/2020/1360832",
    }
    joblib.dump(artifact, model_path)
    return metrics


def load_model(path: str | Path) -> dict[str, object]:
    artifact = joblib.load(path)
    required = {"classifier", "segmentation_config", "feature_names", "class_names"}
    if not isinstance(artifact, dict) or not required.issubset(artifact):
        raise ValueError("Not a valid paper implementation model artifact")
    return artifact


def predict_image(image: np.ndarray, artifact: dict[str, object]) -> tuple[int, str, np.ndarray, object]:
    config = SegmentationConfig(**artifact["segmentation_config"])
    result = PaperSegmenter(config).segment(image)
    if not result.is_defect:
        return 0, "nondefect", np.zeros(len(FEATURE_NAMES)), result
    features = extract_features(result.all_components, result.mask)
    label = int(artifact["classifier"].predict(features.reshape(1, -1))[0])
    return label, CLASS_NAMES[label], features, result

