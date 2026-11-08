"""Evaluate a trained checkpoint exactly once on a held-out test split."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader

from tomatolab.data import load_split
from tomatolab.models import build_model


def load_checkpoint(path, device):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_type"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device).eval()
    return model, checkpoint


def evaluate(model, loader, device, class_names):
    actual, predicted = [], []
    with torch.inference_mode():
        for images, labels in loader:
            logits = model(images.to(device))
            actual.extend(labels.tolist())
            predicted.extend(logits.argmax(1).cpu().tolist())
    ids = list(range(len(class_names)))
    return {
        "n_samples": len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, labels=ids, average="macro", zero_division=0)),
        "classification_report": classification_report(actual, predicted, labels=ids, target_names=class_names, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(actual, predicted, labels=ids).tolist(),
        "class_names": class_names,
    }


def save_confusion_matrix(result, destination):
    matrix = np.asarray(result["confusion_matrix"])
    labels = [x.replace("Tomato___", "").replace("_", " ") for x in result["class_names"]]
    fig, ax = plt.subplots(figsize=(10, 9))
    image = ax.imshow(matrix, cmap="Blues")
    fig.colorbar(image, ax=ax)
    ax.set_xticks(range(len(labels)), labels=labels, rotation=60, ha="right")
    ax.set_yticks(range(len(labels)), labels=labels)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Test confusion matrix (counts)")
    fig.tight_layout()
    fig.savefig(destination, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(args.checkpoint, device)
    test_set = load_split(args.data_dir, "test")
    if test_set.classes != checkpoint["class_names"]:
        raise ValueError("Test dataset class order differs from the trained checkpoint.")
    loader = DataLoader(test_set, batch_size=args.batch_size, num_workers=args.num_workers)
    results = evaluate(model, loader, device, test_set.classes)
    output = Path(args.checkpoint).parent
    (output / "test_metrics.json").write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    save_confusion_matrix(results, output / "test_confusion_matrix.png")
    print(f"Test accuracy: {results['accuracy']:.4f} | Macro F1: {results['macro_f1']:.4f}")
    print(f"Saved metrics and confusion matrix in {output}")


if __name__ == "__main__":
    main()
