"""Segmentation measures used in the paper."""

from __future__ import annotations

import numpy as np


def physical_area_cm2(
    pixel_area: float, reference_length_cm: float, reference_length_pixels: float
) -> float:
    """Equations (4)-(5): convert a pixel area using a scale visible in the image."""
    if reference_length_cm <= 0 or reference_length_pixels <= 0:
        raise ValueError("Reference lengths must be positive")
    cm_per_pixel = reference_length_cm / reference_length_pixels
    return float(pixel_area) * cm_per_pixel**2


def mask_metrics(prediction: np.ndarray, reference: np.ndarray) -> dict[str, float]:
    predicted = prediction > 0
    truth = reference > 0
    intersection = int(np.count_nonzero(predicted & truth))
    union = int(np.count_nonzero(predicted | truth))
    truth_area = int(np.count_nonzero(truth))
    predicted_area = int(np.count_nonzero(predicted))
    return {
        # The paper's "overlap rate" uses the manually measured pothole as denominator.
        "paper_overlap_rate": intersection / truth_area if truth_area else 0.0,
        "iou": intersection / union if union else 1.0,
        "area_extraction_rate": predicted_area / truth_area if truth_area else 0.0,
    }

