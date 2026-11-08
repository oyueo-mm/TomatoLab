# TomatoLab

토마토 잎 이미지 기반 질병 분류 및 인공지능 모델 비교·시각화 실험실

## 프로젝트 소개
PlantVillage의 color 이미지에서 토마토 잎 10개 클래스(정상 포함)를 분류하고 기본 CNN과 ResNet18의 성능을 비교합니다. 정확도·Macro F1·클래스별 Recall·혼동행렬을 분석하며 학습된 체크포인트로 Gradio 웹 데모를 실행합니다.

> 초기 버전: 코드만 준비되었습니다. 데이터셋과 학습된 모델은 아직 포함되지 않았으며 검증된 성능 수치를 주장하지 않습니다.

## 데이터
- [PlantVillage 데이터셋 (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
- 압축 해제 후 **color** 하위의 `Tomato___...` 폴더 10개가 한 곳에 있는 디렉터리를 지정하세요.
- **grayscale**, **segmented** 이미지와 합치지 않습니다.
- 데이터는 Git 저장소에 업로드하지 않습니다.

## 시작하기 (Python 3.10+, Colab GPU 권장)
1. 환경 설치: `pip install -r requirements.txt`
2. 데이터 준비: `python scripts/prepare_data.py --source /path/to/PlantVillage/color --output data/split`
3. 기본 CNN 학습: `python train.py --model cnn --epochs 10`
4. ResNet18 전이학습: `python train.py --model resnet18 --epochs 10`
5. 홀드아웃 평가: `python evaluate.py --checkpoint artifacts/resnet18/best.pt`
6. 로컬 웹 데모: `python app.py --checkpoint artifacts/resnet18/best.pt`

ResNet18의 ImageNet 사전학습 가중치는 처음 실행할 때 다운로드될 수 있습니다.

## 구조
| 경로 | 역할 |
|---|---|
| `scripts/prepare_data.py` | 원본 이미지 확인, SHA-256 정확 중복 제거, 클래스별 70/15/15 분할 |
| `tomatolab/models.py` | 기본 CNN · ResNet18 |
| `tomatolab/data.py` | 로더·전처리·정규화 |
| `train.py` | 학습 및 검증 Macro F1 최고 체크포인트 저장 |
| `evaluate.py` | 테스트 정확도, Macro F1, 클래스별 평가, 혼동행렬 PNG |
| `app.py` | 실제 가중치로 작동하는 Gradio 웹앱 |
| `tests/` | 간단한 자동 검증 |

## 평가 원칙
클래스별 균형을 위해 손실에 빈도의 역제곱근 가중치를 적용합니다. 체크포인트는 **검증 Macro F1**로 선택하고, 테스트셋은 별도 평가에 사용합니다. 같은 파일의 **완전 중복은 제거**하지만, 유사 이미지나 같은 잎에서 촬영된 복수 사진의 누수 방지는 보장하지 않습니다.

### 한계 및 주의
PlantVillage는 통제된 배경 이미지가 많습니다. 실외 사진에서의 일반화 성능은 검증되지 않았으며 실제 병해 진단에 사용하면 안 됩니다. 웹 데모의 softmax 값은 임상적·농업적 신뢰도 점수가 아닙니다.

## 후속 확장
Colab 실행 노트북, 학습 곡선, Grad-CAM, 외부 실환경 이미지 테스트, 정적 웹 데모.

## 참고 문서
- [torchvision ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html)
- [PlantVillage (Kaggle)](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)
