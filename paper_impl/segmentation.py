"""Texture/grayscale/morphology segmentation described in Gao et al. (2020).

The source paper states the processing stages but does not publish the entropy
window, binarization threshold, or morphology element. Defaults here follow
common MATLAB behavior (a 9x9 entropy neighborhood and Otsu thresholding) and
all unpublished choices are configurable.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np


@dataclass(frozen=True)
class SegmentationConfig:
    entropy_window: int = 9
    threshold: float | None = None
    polarity: Literal["auto", "normal", "inverted"] = "auto"
    min_object_area: int = 1500
    close_kernel: int = 9
    close_iterations: int = 1
    keep_largest: bool = True
    nondefect_ratio: float = 0.03

    def __post_init__(self) -> None:
        if self.entropy_window < 1 or self.entropy_window % 2 == 0:
            raise ValueError("entropy_window must be a positive odd integer")
        if self.threshold is not None and not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        if self.polarity not in {"auto", "normal", "inverted"}:
            raise ValueError("polarity must be auto, normal, or inverted")
        if self.min_object_area < 0:
            raise ValueError("min_object_area must be non-negative")
        if self.close_kernel < 1 or self.close_kernel % 2 == 0:
            raise ValueError("close_kernel must be a positive odd integer")
        if not 0.0 <= self.nondefect_ratio <= 1.0:
            raise ValueError("nondefect_ratio must be in [0, 1]")


@dataclass
class SegmentationResult:
    gray: np.ndarray
    entropy: np.ndarray
    entropy_gray: np.ndarray
    binary: np.ndarray
    area_filtered: np.ndarray
    morph_closed: np.ndarray
    all_components: np.ndarray
    mask: np.ndarray
    overlay: np.ndarray
    foreground_ratio: float
    is_defect: bool
    threshold_value: float
    polarity: str


def read_image(path: str | Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read image: {path}")
    return image


def local_entropy(gray: np.ndarray, window: int = 9) -> np.ndarray:
    """Compute Shannon entropy in a square neighborhood for an 8-bit image.

    This is the OpenCV equivalent of MATLAB ``entropyfilt`` for a square
    neighborhood. A per-intensity box filter avoids materializing all sliding
    windows and keeps memory usage bounded.
    """
    if gray.ndim != 2 or gray.dtype != np.uint8:
        raise ValueError("local_entropy expects a two-dimensional uint8 image")
    if window < 1 or window % 2 == 0:
        raise ValueError("window must be a positive odd integer")

    entropy = np.zeros(gray.shape, dtype=np.float32)
    area = float(window * window)
    for value in np.unique(gray):
        indicator = (gray == value).astype(np.float32)
        counts = cv2.boxFilter(
            indicator,
            ddepth=cv2.CV_32F,
            ksize=(window, window),
            normalize=False,
            borderType=cv2.BORDER_REFLECT,
        )
        probability = counts / area
        positive = probability > 0
        entropy[positive] -= probability[positive] * np.log2(probability[positive])
    return entropy


def normalize_u8(image: np.ndarray) -> np.ndarray:
    minimum = float(np.min(image))
    maximum = float(np.max(image))
    if maximum <= minimum:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = (image.astype(np.float32) - minimum) * (255.0 / (maximum - minimum))
    return np.rint(scaled).astype(np.uint8)


def remove_small_components(mask: np.ndarray, minimum_area: int) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        if int(stats[label, cv2.CC_STAT_AREA]) >= minimum_area:
            output[labels == label] = 255
    return output


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill background regions that do not touch the image border."""
    foreground = mask > 0
    background = (~foreground).astype(np.uint8)
    count, labels = cv2.connectedComponents(background, connectivity=8)
    border_labels = np.unique(
        np.concatenate((labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]))
    )
    holes = np.ones_like(foreground, dtype=bool)
    for label in border_labels:
        holes[labels == label] = False
    # label 0 is the original foreground, not a background hole.
    holes[labels == 0] = False
    return np.where(foreground | holes, 255, 0).astype(np.uint8)


def largest_component(mask: np.ndarray) -> np.ndarray:
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 1:
        return np.zeros_like(mask, dtype=np.uint8)
    label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.where(labels == label, 255, 0).astype(np.uint8)


def largest_interior_component(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the largest component not connected to an image border."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    candidates: list[tuple[int, int]] = []
    for label in range(1, count):
        x, y, w, h, area = map(int, stats[label])
        touches_border = x == 0 or y == 0 or x + w == width or y + h == height
        if not touches_border:
            candidates.append((area, label))
    if not candidates:
        return np.zeros_like(mask, dtype=np.uint8), 0
    area, label = max(candidates)
    return np.where(labels == label, 255, 0).astype(np.uint8), area


def remove_border_components(mask: np.ndarray) -> np.ndarray:
    """Remove connected foreground regions that merge with an image border."""
    binary = (mask > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    height, width = binary.shape
    output = np.zeros_like(mask, dtype=np.uint8)
    for label in range(1, count):
        x, y, w, h, _ = map(int, stats[label])
        if x > 0 and y > 0 and x + w < width and y + h < height:
            output[labels == label] = 255
    return output


def make_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    green = np.zeros_like(image)
    green[:, :, 1] = 255
    selected = mask > 0
    overlay[selected] = cv2.addWeighted(image, 0.55, green, 0.45, 0)[selected]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 0, 255), 2)
    return overlay


class PaperSegmenter:
    def __init__(self, config: SegmentationConfig | None = None) -> None:
        self.config = config or SegmentationConfig()

    def segment(self, image: np.ndarray) -> SegmentationResult:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("segment expects a BGR image with three channels")

        cfg = self.config
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        entropy = local_entropy(gray, cfg.entropy_window)
        entropy_gray = normalize_u8(entropy)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel)
        )

        def process_candidate(threshold_value: int, polarity: str) -> tuple[np.ndarray, ...]:
            threshold_type = cv2.THRESH_BINARY if polarity == "normal" else cv2.THRESH_BINARY_INV
            _, candidate_binary = cv2.threshold(
                entropy_gray, threshold_value, 255, threshold_type
            )
            candidate_filtered = remove_small_components(candidate_binary, cfg.min_object_area)
            candidate_closed = cv2.morphologyEx(
                candidate_filtered,
                cv2.MORPH_CLOSE,
                kernel,
                iterations=cfg.close_iterations,
            )
            candidate_components = fill_holes(candidate_closed)
            candidate_interior, candidate_area = largest_interior_component(candidate_components)
            return (
                candidate_binary,
                candidate_filtered,
                candidate_closed,
                candidate_components,
                candidate_interior,
                candidate_area,
            )

        if cfg.threshold is not None:
            threshold_values = [int(round(cfg.threshold * 255))]
        elif cfg.polarity == "auto":
            otsu_value, _ = cv2.threshold(
                entropy_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            percentiles = np.percentile(entropy_gray, np.arange(55, 96, 5))
            threshold_values = sorted({int(round(otsu_value)), *(int(round(v)) for v in percentiles)})
        else:
            otsu_value, _ = cv2.threshold(
                entropy_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            threshold_values = [int(round(otsu_value))]

        polarities = [cfg.polarity] if cfg.polarity != "auto" else ["normal", "inverted"]
        candidates: list[tuple[int, int, str, tuple[np.ndarray, ...]]] = []
        for threshold_value in threshold_values:
            for polarity in polarities:
                stages = process_candidate(threshold_value, polarity)
                candidates.append((int(stages[-1]), threshold_value, polarity, stages))

        # A pothole is an enclosed pavement region. Prefer the largest candidate
        # that does not merge with an image border. If none exists, retain the
        # ordinary largest domain so explicit/manual workflows still yield output.
        interior_area, chosen_threshold, chosen_polarity, chosen = max(
            candidates, key=lambda candidate: candidate[0]
        )
        binary, area_filtered, morph_closed, all_components, interior_mask, _ = chosen
        if cfg.polarity == "auto" and interior_area > 0:
            all_components = remove_border_components(all_components)
        if cfg.keep_largest:
            mask = largest_component(all_components)
        else:
            mask = all_components.copy()
        foreground_ratio = float(np.count_nonzero(all_components)) / float(all_components.size)

        return SegmentationResult(
            gray=gray,
            entropy=entropy,
            entropy_gray=entropy_gray,
            binary=binary,
            area_filtered=area_filtered,
            morph_closed=morph_closed,
            all_components=all_components,
            mask=mask,
            overlay=make_overlay(image, mask),
            foreground_ratio=foreground_ratio,
            is_defect=foreground_ratio >= cfg.nondefect_ratio,
            threshold_value=chosen_threshold / 255.0,
            polarity=chosen_polarity,
        )

