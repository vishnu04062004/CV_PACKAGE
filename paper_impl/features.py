"""Eight handcrafted features specified by Gao et al. (2020)."""

from __future__ import annotations

import cv2
import numpy as np


FEATURE_NAMES = (
    "connected_domains",
    "projection_ratio",
    "fractal_dimension",
    "area",
    "perimeter",
    "circularity",
    "rectangle_degree",
    "aspect_ratio",
)


def _box_count_fractal_dimension(mask: np.ndarray) -> float:
    binary = mask.astype(bool)
    height, width = binary.shape
    maximum_power = int(np.floor(np.log2(min(height, width))))
    sizes: list[int] = []
    counts: list[int] = []

    for power in range(maximum_power, 0, -1):
        size = 2**power
        padded_h = int(np.ceil(height / size) * size)
        padded_w = int(np.ceil(width / size) * size)
        padded = np.zeros((padded_h, padded_w), dtype=bool)
        padded[:height, :width] = binary
        blocks = padded.reshape(padded_h // size, size, padded_w // size, size)
        count = int(np.count_nonzero(blocks.any(axis=(1, 3))))
        if count > 0:
            sizes.append(size)
            counts.append(count)

    if len(counts) < 2:
        return 0.0
    slope = np.polyfit(np.log(1.0 / np.asarray(sizes)), np.log(counts), 1)[0]
    return float(slope)


def extract_features(all_components: np.ndarray, primary_mask: np.ndarray | None = None) -> np.ndarray:
    """Return the paper's eight features in ``FEATURE_NAMES`` order.

    The connected-domain feature is measured before maximum-domain extraction;
    geometric features are measured on the maximum domain, matching the paper's
    descriptions of both operations.
    """
    binary_all = (all_components > 0).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary_all, connectivity=8)
    connected_domains = max(0, count - 1)

    if primary_mask is None:
        if connected_domains == 0:
            primary = np.zeros_like(binary_all)
        else:
            label = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            primary = (labels == label).astype(np.uint8)
    else:
        primary = (primary_mask > 0).astype(np.uint8)

    area = float(np.count_nonzero(primary))
    if area == 0:
        return np.array([connected_domains, 0, 0, 0, 0, 0, 0, 0], dtype=np.float64)

    # Equation (11): range of the column projection divided by range of the row projection.
    x_projection = primary.sum(axis=0).astype(np.float64)
    y_projection = primary.sum(axis=1).astype(np.float64)
    x_range = float(x_projection.max() - x_projection.min())
    y_range = float(y_projection.max() - y_projection.min())
    projection_ratio = x_range / y_range if y_range > 0 else 0.0

    fractal_dimension = _box_count_fractal_dimension(primary)
    contours, _ = cv2.findContours(primary * 255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    perimeter = float(sum(cv2.arcLength(contour, True) for contour in contours))
    circularity = 4.0 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0.0

    points = cv2.findNonZero(primary)
    (_, _), (rect_w, rect_h), _ = cv2.minAreaRect(points)
    rectangle_area = float(rect_w * rect_h)
    rectangle_degree = area / rectangle_area if rectangle_area > 0 else 0.0
    short_side = min(float(rect_w), float(rect_h))
    long_side = max(float(rect_w), float(rect_h))
    aspect_ratio = long_side / short_side if short_side > 0 else 0.0

    return np.array(
        [
            connected_domains,
            projection_ratio,
            fractal_dimension,
            area,
            perimeter,
            circularity,
            rectangle_degree,
            aspect_ratio,
        ],
        dtype=np.float64,
    )

