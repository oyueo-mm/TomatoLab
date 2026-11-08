"""Gradio inference demo; real predictions require a trained checkpoint."""
import argparse
from pathlib import Path

import gradio as gr
import torch

from evaluate import load_checkpoint
from tomatolab.data import image_transform


def readable_name(name):
    return name.replace("Tomato___", "").replace("_", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/resnet18/best.pt")
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    checkpoint_path = Path(args.checkpoint)
    available = checkpoint_path.is_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, checkpoint = load_checkpoint(checkpoint_path, device) if available else (None, None)
    preprocess = image_transform(training=False)

    def predict(image):
        if image is None:
            raise gr.Error("토마토 잎 이미지를 업로드해 주세요.")
        with torch.inference_mode():
            batch = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            probabilities = torch.softmax(model(batch), dim=1)[0].cpu().tolist()
        return {readable_name(label): float(prob) for label, prob in zip(checkpoint["class_names"], probabilities)}

    with gr.Blocks(title="TomatoLab | 토마토 잎 분류 실험실") as demo:
        gr.Markdown("# TomatoLab\n토마토 잎 이미지 기반 딥러닝 분류 실험실")
        if available:
            gr.Markdown(f"현재 모델: **{checkpoint['model_type']}** · {len(checkpoint['class_names'])}개 클래스\n\n**주의:** 확률은 모델의 예측값이며 실제 질병 확진이나 현장 성능을 의미하지 않습니다.")
            image = gr.Image(type="pil", sources=["upload", "webcam"], label="토마토 잎 사진")
            output = gr.Label(num_top_classes=5, label="모델 예측 확률")
            gr.Button("이미지 분석", variant="primary").click(predict, inputs=image, outputs=output)
        else:
            gr.Markdown(f"**아직 학습된 모델이 없습니다.** \n\n먼저 학습을 완료한 뒤 체크포인트를 준비해 주세요: \n\n{checkpoint_path}\n\n이 화면은 가짜 분류 결과를 제공하지 않습니다.")
    demo.launch(share=args.share)


if __name__ == "__main__":
    main()
