# TomatoLab

토마토 잎 사진으로 질병을 분류하고, 딥러닝 모델의 성능을 비교·시각화하는 실험 프로젝트입니다.

## 목표
- PlantVillage의 토마토 잎 10개 범주(정상 포함)를 분류
- 직접 구현한 CNN과 ImageNet 사전학습 ResNet18 비교
- Accuracy, Macro F1, 클래스별 Recall, 혼동행렬 분석
- 학습된 가중치로 사진을 분석하는 Gradio 데모 제공

> **현재 상태:** 학습 파이프라인과 데모 코드가 준비된 초기 버전입니다. 데이터와 학습된 모델 가중치는 아직 저장소에 포함되어 있지 않습니다. 검증된 성능 수치를 주장하지 않습니다.

## 데이터
[PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)의 **color** 폴더에 있는 토마토 10개 클래스를 사용합니다. 다운로드한 데이터에서 `Tomato___...` 폴더 10개가 보이는 위치를 `--source`로 지정하세요. **grayscale/segmented 폴더를 섞지 마세요.**

원본 이미지는 저장소에 올리지 않습니다. 준비 스크립트는 정확히 동일한 이미지의 SHA-256 중복을 제거하고 클래스별 70/15/15 비율로 고정 시드 분할합니다. 유사 이미지·같은 잎의 여러 사진에 대한 완전한 누수 방지는 보장되지 않습니다.

## 빠른 시작 (Python 3.10+, Colab GPU 권장)
1. `pip install -r requirements.txt`
2. 데이터셋을 내려받아 압축을 풀고 `Tomato___...` 클래스 폴더가 모인 경로를 확인합니다.
3. `python scripts/prepare_data.py --source /경로/PlantVillage/color --output data/split`
4. `python train.py --model cnn --data-dir data/split --epochs 10`
5. `python train.py --model resnet18 --data-dir data/split --epochs 10`
6. `python evaluate.py --checkpoint artifacts/resnet18/best.pt --data-dir data/split`
7. `python app.py --checkpoint artifacts/resnet18/best.pt`

학습 시 처음 사용되는 ResNet18 사전학습 가중치는 네트워크 연결이 필요합니다. 로컬 GPU가 없으면 학습이 느릴 수 있으니 Google Colab에서 실행하세요.

## 파일 구성
- `scripts/prepare_data.py`: 원본 무결성 확인, 중복 제거, 계층화 분할
- `tomatolab/models.py`: CNN / ResNet18 모델 정의
- `tomatolab/data.py`: 데이터로더 및 학습·평가 이미지 변환
- `train.py`: 학습, 검증, 최고 모델 저장 및 이력 기록
- `evaluate.py`: 홀드아웃 테스트, 지표와 혼동행렬 출력
- `app.py`: 학습된 모델을 불러오는 Gradio 이미지 분류 UI
- `tests/`: 최소 기능 자동 테스트

## 평가 설계
검증셋의 **Macro F1**이 가장 높은 체크포인트를 선택하고, 테스트셋은 마지막에 별도로 평가합니다. 클래스 분포와 클래스별 Recall도 확인해야 하며, 정확도만으로 질병 분류 성능을 판단하지 않습니다.

**주의:** PlantVillage는 통제된 환경에서 촬영한 이미지가 많습니다. 실외 사진에 대한 일반화 성능과 실제 병해 진단의 신뢰성을 확인하지 않았으므로 농업 의사결정이나 실제 진단 목적으로 사용하지 마세요.

## 다음 단계
Colab 실습 노트북, 학습 결과 시각화 개선, Grad-CAM, 실제 환경의 외부 테스트셋, 정적 웹 배포 방식을 순차적으로 추가할 예정입니다.

## 참고
- [PlantVillage dataset](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- [torchvision ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)

think about it step-by-step.
