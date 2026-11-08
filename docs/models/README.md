# TomatoLab browser model

이 폴더에 Colab에서 내보낸 **resnet18.onnx**를 넣으세요.

- 경로: `docs/models/resnet18.onnx`
- 이 파일이 배포되면 `docs/config.json`의 `modelUrl` 설정에 따라 GitHub Pages가 브라우저에서 모델을 자동 다운로드합니다.
- 업로드 대상 모델: ResNet18 / ONNX opset 17 / 입력 `[1, 3, 224, 224]` / 출력 `[1, 10]`
- 업로드 파일 크기: 44,717,066 bytes (약 42.6 MiB)
- 사용자가 2026-11-08 업로드한 검증 파일 SHA-256: `1023badd0d7f65335afc68939e2d7d4b665fbee64778a6bf0929bdaf0ad5952d`

GitHub 웹 UI의 파일 크기 제한 때문에 이 크기의 파일은 GitHub Desktop 또는 로컬 Git으로 푸시하세요. `*.onnx`는 기본적으로 `.gitignore`에 포함되지만, `docs/models/*.onnx`는 예외 설정되어 있습니다.

외부 농업 현장 사진에서의 정확도는 검증되지 않았으므로 실제 병해 진단에 사용하지 마세요.