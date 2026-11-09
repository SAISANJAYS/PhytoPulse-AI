import sys
import unittest
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from PIL import Image
import numpy as np

from src.config import MODEL_PATH, CLASS_NAMES_PATH
from src.image_predictor import LeafDiseasePredictor


class TestLeafDiseasePredictor(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.predictor = LeafDiseasePredictor(
            model_path=MODEL_PATH,
            class_names_path=CLASS_NAMES_PATH,
        )

    def test_model_loaded(self):
        self.assertIsNotNone(self.predictor.model)
        self.assertEqual(len(self.predictor.class_names), 4)
        self.assertEqual(int(self.predictor.model.output_shape[-1]), 4)

    def test_predict_synthetic_image(self):
        # Create a synthetic 512x512 RGB PIL Image
        img = Image.new("RGB", (512, 512), color=(120, 180, 70))
        result = self.predictor.predict(img)

        self.assertIn("class_name", result)
        self.assertIn("confidence", result)
        self.assertIn("all_scores", result)

        self.assertIn(result["class_name"], self.predictor.class_names)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)

        # Check all_scores
        self.assertEqual(len(result["all_scores"]), 4)
        scores = [item["score"] for item in result["all_scores"]]
        self.assertEqual(scores, sorted(scores, reverse=True))
        self.assertAlmostEqual(sum(scores), 1.0, places=4)

    def test_predict_real_image(self):
        test_dir = Path("data/test_images/disease/segmented")
        sample_images = list(test_dir.glob("*.jpg"))
        if not sample_images:
            self.skipTest("No test images found in data/test_images")

        img = Image.open(sample_images[0])
        result = self.predictor.predict(img)

        self.assertIn(result["class_name"], self.predictor.class_names)
        self.assertGreaterEqual(result["confidence"], 0.0)
        self.assertLessEqual(result["confidence"], 1.0)


if __name__ == "__main__":
    unittest.main()
