# TomatoLab 정적 웹사이트

GitHub Pages는 Python 모델을 서버에서 실행할 수 없으므로 ONNX Runtime Web을 통해 브라우저에서 분류합니다.

## 현재 상태

- 웹 화면, 이미지 업로드, 실제 ONNX 추론 로직, 모델 비교·Recall·혼동행렬 구현 완료
- **실제 .onnx 모델 파일은 아직 저장소에 없습니다.** 이 파일이 없으면 사진을 분석할 수 없습니다.
- 방문자가 직접 .onnx 파일을 선택하면 서버로 사진을 보내지 않고 로컬 브라우저에서 실행합니다.

## 1. Colab / Python에서 모델 내보내기

설치: `pip install onnx onnxruntime`

변환: `python scripts/export_onnx.py --checkpoint artifacts/resnet18/best.pt --output docs/models/resnet18.onnx`

스크립트는 PyTorch와 ONNX Runtime 추론 출력 일치 여부를 확인한 뒤 모델과 메타데이터를 저장합니다.

## 2. 웹사이트에 모델 연결

단일 사용자 실험: 웹사이트에서 `.onnx` 모델 파일을 직접 선택합니다.

모든 방문자에게 자동 제공하려면 `docs/models/resnet18.onnx`를 저장소에 푸시하고 `docs/config.json`의 `modelUrl`을 `./models/resnet18.onnx`로 수정하세요.

ONNX 파일은 수십 MB로 예상됩니다. GitHub 브라우저 업로드는 파일당 25MiB 제한이고, Git CLI는 100MiB까지 허용합니다. GitHub Desktop 또는 Git CLI를 사용하거나 별도 CORS 지원 호스팅에 올리세요.

## 3. GitHub Pages 활성화

저장소 Settings → Pages → Build and deployment → Source에서 **GitHub Actions**를 선택합니다.

사이트 주소는 활성화 후 GitHub Pages 설정에서 확인하세요. 배포 전까지 URL이 작동한다고 보장하지 않습니다.

> PlantVillage는 통제된 촬영 환경의 이미지가 많으므로, 이 데모는 농업 진단용이 아닙니다.