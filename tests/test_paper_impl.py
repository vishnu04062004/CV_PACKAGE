import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import cv2
import numpy as np

from paper_impl.features import FEATURE_NAMES, extract_features
from paper_impl.metrics import mask_metrics, physical_area_cm2
from paper_impl.model import load_model, train_model
from paper_impl.segmentation import (
    PaperSegmenter,
    SegmentationConfig,
    fill_holes,
    local_entropy,
    remove_small_components,
)


class PaperImplementationTests(unittest.TestCase):
    def test_entropy_is_zero_for_constant_image(self):
        gray = np.full((31, 31), 127, dtype=np.uint8)
        self.assertTrue(np.allclose(local_entropy(gray), 0.0))

    def test_small_components_and_hole_filling(self):
        mask = np.zeros((40, 40), dtype=np.uint8)
        mask[2:4, 2:4] = 255
        mask[10:30, 10:30] = 255
        mask[15:20, 15:20] = 0
        cleaned = remove_small_components(mask, 100)
        filled = fill_holes(cleaned)
        self.assertEqual(int(cleaned[2, 2]), 0)
        self.assertEqual(int(filled[17, 17]), 255)

    def test_eight_features_for_rectangle(self):
        mask = np.zeros((100, 120), dtype=np.uint8)
        mask[20:80, 30:90] = 255
        values = extract_features(mask, mask)
        self.assertEqual(len(values), len(FEATURE_NAMES))
        self.assertEqual(values[0], 1)
        self.assertAlmostEqual(values[3], 3600)
        self.assertGreater(values[5], 0)
        self.assertGreater(values[6], 0.9)

    def test_segmenter_returns_all_paper_stages(self):
        image = np.full((160, 160, 3), 150, dtype=np.uint8)
        rng = np.random.default_rng(4)
        image[45:120, 45:120] = rng.integers(20, 220, (75, 75, 3), dtype=np.uint8)
        result = PaperSegmenter(
            SegmentationConfig(min_object_area=50, close_kernel=3)
        ).segment(image)
        self.assertEqual(result.mask.shape, image.shape[:2])
        self.assertGreater(np.count_nonzero(result.mask), 0)

    def test_mask_metrics(self):
        truth = np.zeros((10, 10), dtype=np.uint8)
        truth[2:6, 2:6] = 255
        pred = truth.copy()
        values = mask_metrics(pred, truth)
        self.assertEqual(values["paper_overlap_rate"], 1.0)
        self.assertEqual(values["iou"], 1.0)

    def test_physical_area_conversion(self):
        # A 10 cm reference spanning 100 pixels gives 0.01 cm2 per pixel.
        self.assertAlmostEqual(physical_area_cm2(5000, 10, 100), 50.0)

    def test_four_class_training_artifact(self):
        # Four separated feature clusters, with four samples per paper class.
        x = np.vstack(
            [np.full((4, len(FEATURE_NAMES)), label * 10.0) for label in range(1, 5)]
        )
        y = np.repeat(np.arange(1, 5), 4)
        paths = [f"sample-{index}.png" for index in range(len(y))]
        with TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.joblib"
            with patch("paper_impl.model.build_feature_dataset", return_value=(x, y, paths)):
                metrics = train_model("unused", model_path)
            artifact = load_model(model_path)
            self.assertIn("pothole_f1", metrics)
            self.assertEqual(tuple(artifact["feature_names"]), FEATURE_NAMES)


if __name__ == "__main__":
    unittest.main()

