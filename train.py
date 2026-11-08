"""Train a CNN or ResNet18; select the best checkpoint by validation Macro F1."""
import argparse
import csv
import json
import math
import random
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader

from tomatolab.data import IMAGE_SIZE, MEAN, STD, check_same_classes, load_split
from tomatolab.models import build_model


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    actual, predicted = [], []
    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.size(0)
        actual.extend(labels.detach().cpu().tolist())
        predicted.extend(logits.detach().argmax(1).cpu().tolist())
    if not actual:
        raise ValueError("Empty dataset split.")
    return {
        "loss": total_loss / len(actual),
        "accuracy": float(accuracy_score(actual, predicted)),
        "macro_f1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["cnn", "resnet18"], default="resnet18")
    parser.add_argument("--data-dir", default="data/split")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive.")
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_set = load_split(args.data_dir, "train", training=True)
    val_set = load_split(args.data_dir, "val")
    class_names = check_same_classes(train_set, val_set)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=device.type == "cuda", generator=generator)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                            pin_memory=device.type == "cuda")
    counts = Counter(label for _, label in train_set.samples)
    class_weights = torch.tensor([1 / math.sqrt(counts[i]) for i in range(len(class_names))], dtype=torch.float)
    class_weights /= class_weights.mean()
    model = build_model(args.model, len(class_names), pretrained=args.model == "resnet18" and not args.no_pretrained).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    output = Path(args.output_dir or f"artifacts/{args.model}")
    output.mkdir(parents=True, exist_ok=True)
    history, best = [], -1.0
    for epoch in range(1, args.epochs + 1):
        train_stats = run_epoch(model, train_loader, criterion, device, optimizer)
        val_stats = run_epoch(model, val_loader, criterion, device)
        record = {"epoch": epoch, **{f"train_{k}": v for k, v in train_stats.items()},
                  **{f"val_{k}": v for k, v in val_stats.items()}}
        history.append(record)
        print(f"Epoch {epoch}/{args.epochs} | train F1={train_stats['macro_f1']:.4f} | val F1={val_stats['macro_f1']:.4f} | val acc={val_stats['accuracy']:.4f}")
        if val_stats["macro_f1"] > best:
            best = val_stats["macro_f1"]
            torch.save({
                "model_type": args.model, "state_dict": model.state_dict(), "class_names": class_names,
                "epoch": epoch, "val_macro_f1": best, "image_size": IMAGE_SIZE, "mean": list(MEAN),
                "std": list(STD), "seed": args.seed,
            }, output / "best.pt")
    with (output / "history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    (output / "run_config.json").write_text(json.dumps(vars(args), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Best validation Macro F1: {best:.4f}; checkpoint: {output / 'best.pt'}")


if __name__ == "__main__":
    main()
