"""Export a trained TomatoLab PyTorch checkpoint for browser inference.

Install: pip install torch torchvision onnx onnxruntime
Run: python scripts/export_onnx.py --checkpoint artifacts/resnet18/best.pt --output docs/models/resnet18.onnx
"""
import argparse
import json
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

from tomatolab.models import build_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/resnet18/best.pt")
    parser.add_argument("--output", default="docs/models/resnet18.onnx")
    args = parser.parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = build_model(checkpoint["model_type"], len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    dummy = torch.randn(1, 3, 224, 224)
    with torch.inference_mode():
        expected = model(dummy).numpy()
        torch.onnx.export(
            model, (dummy,), str(destination),
            input_names=["input"], output_names=["logits"],
            opset_version=17, dynamo=False, do_constant_folding=True,
        )
    onnx.checker.check_model(str(destination))
    session = ort.InferenceSession(str(destination), providers=["CPUExecutionProvider"])
    actual = session.run(["logits"], {"input": dummy.numpy()})[0]
    np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
    manifest = {
        "model_type": checkpoint["model_type"],
        "class_names": checkpoint["class_names"],
        "input_size": [1, 3, 224, 224],
        "mean": checkpoint["mean"],
        "std": checkpoint["std"],
        "preprocessing": "resize(256,256) -> center crop(224) -> RGB /255 -> ImageNet normalize",
        "validation_macro_f1": checkpoint["val_macro_f1"],
        "checkpoint_epoch": checkpoint["epoch"],
    }
    destination.with_suffix(".json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ONNX model exported: {destination} ({destination.stat().st_size / (1024*1024):.1f} MiB)")
    print(f"ONNX Runtime/PyTorch logits verified: max diff {np.max(np.abs(expected-actual)):.5g}")


if __name__ == "__main__":
    main()
