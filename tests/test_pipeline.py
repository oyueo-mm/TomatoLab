"""Small smoke tests; no PlantVillage download needed."""
import tempfile
import unittest
from pathlib import Path

import torch
from PIL import Image

from scripts.prepare_data import prepare_dataset
from tomatolab.data import check_same_classes, load_split
from tomatolab.models import build_model


class TomatoLabTests(unittest.TestCase):
    def test_cnn_output_shape(self):
        model = build_model("cnn", num_classes=10)
        model.eval()
        with torch.inference_mode():
            out = model(torch.rand(2, 3, 64, 64))
        self.assertEqual(tuple(out.shape), (2, 10))

    def test_preparation_removes_exact_duplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "color"
            output = Path(temp) / "splits"
            for cls in ("Tomato___healthy", "Tomato___Late_blight"):
                folder = source / cls
                folder.mkdir(parents=True)
                for idx in range(6):
                    Image.new("RGB", (32, 32), ((idx * 25) % 255, 70 if cls.endswith("healthy") else 100, 20)).save(folder / f"{idx}.png")
                (folder / "duplicate.png").write_bytes((folder / "0.png").read_bytes())
            result = prepare_dataset(source, output, seed=42, expected_classes=2)
            self.assertEqual(result["exact_duplicates_removed"], 2)
            train = load_split(str(output), "train")
            val = load_split(str(output), "val")
            test = load_split(str(output), "test")
            self.assertEqual(len(check_same_classes(train, val, test)), 2)
            self.assertEqual(len(train) + len(val) + len(test), 12)
            fingerprints = {}
            for split in ("train", "val", "test"):
                for path in (output / split).rglob("*.png"):
                    raw = path.read_bytes()
                    self.assertNotIn(raw, fingerprints)
                    fingerprints[raw] = split


if __name__ == "__main__":
    unittest.main()
