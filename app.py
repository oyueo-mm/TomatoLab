"""TomatoLab Gradio web application. Real inference requires a trained checkpoint."""
import argparse
import json
from pathlib import Path

import gradio as gr
import torch
from PIL import Image

from tomatolab.data import image_transform
from tomatolab.models import build_model

KR = {
    "Tomato___Bacterial_spot": "세균성 반점병",
    "Tomato___Early_blight": "겹무늬병",
    "Tomato___Late_blight": "역병",
    "Tomato___Leaf_Mold": "잎곰팡이병",
    "Tomato___Septoria_leaf_spot": "셉토리아 잎반점병",
    "Tomato___Spider_mites Two-spotted_spider_mite": "점박이응애 피해",
    "Tomato___Target_Spot": "표적반점병",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "황화잎말림바이러스",
    "Tomato___Tomato_mosaic_virus": "토마토 모자이크 바이러스",
    "Tomato___healthy": "정상",
}


def load_classifier(path: Path, device):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    if ckpt.get("model_type") not in ("cnn", "resnet18"):
        raise ValueError("Unsupported model type")
    classes = ckpt["class_names"]
    if len(classes) != 10 or len(set(classes)) != 10:
        raise ValueError("Expected 10 unique classes.")
    model = build_model(ckpt["model_type"], len(classes), pretrained=False)
    model.load_state_dict(ckpt["state_dict"], strict=True)
    model = model.to(device).eval()
    return model, ckpt


def read_metrics(model_type, checkpoint_path: Path):
    if checkpoint_path.parent.name != model_type:
        return None
    path = checkpoint_path.parent / "test_metrics.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def build_app(checkpoint_path: Path):
    available = checkpoint_path.is_file()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, ckpt = load_classifier(checkpoint_path, device) if available else (None, None)
    preprocess = image_transform(training=False)
    scores = read_metrics(ckpt["model_type"], checkpoint_path) if available else None

    def predict(image):
        if image is None or not isinstance(image, Image.Image):
            raise gr.Error("토마토 잎 이미지를 먼저 업로드해 주세요.")
        with torch.inference_mode():
            input_tensor = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
            probs = torch.softmax(model(input_tensor), dim=1)[0].cpu().tolist()
        labels = {KR.get(c, c): float(p) for c, p in zip(ckpt["class_names"], probs)}
        index = max(range(len(probs)), key=probs.__getitem__)
        top = KR.get(ckpt["class_names"][index], ckpt["class_names"][index])
        summary = (
            f"### 예측 결과 · {top}\n"
            f"**모델 예측 확률 {probs[index]*100:.1f}%**\n\n"
            "> 이 수치는 보정된 신뢰도가 아니라 모델의 출력 확률입니다. "
            "현장 질병 진단용으로 사용하지 마세요."
        )
        return summary, labels

    with gr.Blocks(title="TomatoLab · 토마토 잎 분석 실험실") as demo:
        gr.Markdown("# TomatoLab\n토마토 잎 이미지 기반 딥러닝 분류 실험실")
        gr.Markdown(
            "**PlantVillage 연구용 데모** · 통제된 환경에서 수집된 데이터의 "
            "결과를 실제 농장 사진에 일반화할 수 없습니다."
        )
        with gr.Tabs():
            with gr.Tab("사진 분석"):
                if not available:
                    gr.Markdown(
                        "**학습 모델 미설치** — "
                        f"먼저 실제 체크포인트를 준비해 주세요: {checkpoint_path}\n\n"
                        "학습 전 임의의 예측 결과를 표시하지 않습니다."
                    )
                else:
                    gr.Markdown(
                        f"**연결된 모델:** {ckpt['model_type']} · "
                        f"최고 검증 Macro F1 {ckpt['val_macro_f1']:.4f}"
                    )
                    with gr.Row():
                        with gr.Column():
                            picture = gr.Image(type="pil", image_mode="RGB", sources=["upload", "webcam"], label="잎 이미지")
                            button = gr.Button("사진 분석하기", variant="primary")
                        with gr.Column():
                            summary = gr.Markdown("### 분석 대기 중\n토마토 잎 사진을 선택해 주세요.")
                            labels = gr.Label(num_top_classes=10, label="10개 범주 예측 분포")
                    button.click(predict, inputs=picture, outputs=[summary, labels])
            with gr.Tab("모델 성능"):
                if scores:
                    gr.Markdown(
                        f"### 테스트 정확도 {scores['accuracy']*100:.2f}% · "
                        f"Macro F1 {scores['macro_f1']:.4f}\n"
                        f"테스트 이미지 {scores['n_samples']:,}장. "
                        "동일 데이터셋 외부의 성능은 검증되지 않았습니다."
                    )
                    rows = []
                    for cls in ckpt["class_names"]:
                        item = scores["classification_report"].get(cls, {})
                        rows.append([
                            KR.get(cls, cls), round(item.get("precision", 0), 4),
                            round(item.get("recall", 0), 4),
                            round(item.get("f1-score", 0), 4), int(item.get("support", 0))
                        ])
                    gr.Dataframe(
                        value=rows,
                        headers=["질병 범주", "Precision", "Recall", "F1", "테스트 이미지"],
                        interactive=False,
                        label="질병별 평가",
                    )
                else:
                    gr.Markdown("아직 이 체크포인트의 테스트 결과가 없습니다. evaluate.py로 평가하세요.")
            with gr.Tab("프로젝트 소개"):
                gr.Markdown(
                    "PlantVillage의 토마토 잎 이미지 10개 클래스를 CNN·ResNet18로 학습하고 "
                    "정확도, Macro F1, 질병별 성능을 비교합니다.\n\n"
                    "전처리: 256×256 크기 변경 → 중앙 224×224 크롭 → ImageNet 정규화.\n\n"
                    "[GitHub 소스](https://github.com/oyueo-mm/TomatoLab)"
                )
    return demo


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="artifacts/resnet18/best.pt")
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    build_app(Path(args.checkpoint)).launch(
        server_port=args.port,
        share=args.share,
        show_error=True,
    )


if __name__ == "__main__":
    main()
