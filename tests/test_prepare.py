"""Dependency-light checks for tomato dataset preparation (Pillow only)."""
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from scripts.prepare_data import prepare_dataset


class PrepareDataTests(unittest.TestCase):
    def create_source(self, root):
        source = root / "color"
        for label in ["Tomato___healthy", "Tomato___Early_blight"]:
            folder = source / label
            folder.mkdir(parents=True)
            for index in range(6):
                Image.new("RGB", (24, 24), (index * 25, 30 if "healthy" in label else 120, 55)).save(folder / f"{index}.png")
            (folder / "duplicate.png").write_bytes((folder / "0.png").read_bytes())
        return source

    def test_stratified_split_and_exact_dedup(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_source(root)
            output = root / "out"
            report = prepare_dataset(source, output, seed=42, expected_classes=2)
            self.assertEqual(report["exact_duplicates_removed"], 2)
            all_hashes = set()
            for label, values in report["classes"].items():
                self.assertEqual(values["unique_count"], 6)
                self.assertEqual(values["train"] + values["val"] + values["test"], 6)
                self.assertGreater(values["train"], 0)
                self.assertGreater(values["val"], 0)
                self.assertGreater(values["test"], 0)
            for image in output.rglob("*.png"):
                raw = image.read_bytes()
                self.assertNotIn(raw, all_hashes)
                all_hashes.add(raw)
            self.assertEqual(len(all_hashes), 12)

    def test_refuses_accidental_overwrite(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_source(root)
            output = root / "out"
            prepare_dataset(source, output, expected_classes=2)
            with self.assertRaises(FileExistsError):
                prepare_dataset(source, output, expected_classes=2)

    def test_conflicting_labels_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.create_source(root)
            first = source / "Tomato___healthy" / "1.png"
            second = source / "Tomato___Early_blight" / "1.png"
            second.write_bytes(first.read_bytes())
            with self.assertRaises(ValueError):
                prepare_dataset(source, root / "out", expected_classes=2)


if __name__ == "__main__":
    unittest.main()
