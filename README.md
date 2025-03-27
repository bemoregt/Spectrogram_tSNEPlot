# Spectrogram_tSNEPlot

오디오 파일의 스펙트로그램을 시각화하고 t-SNE로 차원 축소하여 인터랙티브하게 분석하는 도구입니다.

## 개요

이 애플리케이션은 오디오 파일(MP3)을 로드하여 여러 세그먼트로 나누고, 각 세그먼트의 스펙트로그램을 ResNet18 딥러닝 모델을 사용하여 특징 벡터로 추출한 후, t-SNE 알고리즘을 통해 2차원으로 시각화합니다. 사용자는 t-SNE 플롯에서 점을 클릭하여 해당 오디오 세그먼트의 스펙트로그램을 보고 소리를 들을 수 있습니다.

## 주요 기능

- MP3 오디오 파일 로드 및 세그먼트 분할
- 각 세그먼트의 스펙트로그램 생성
- ResNet18을 이용한 딥 특징 추출
- t-SNE를 통한 차원 축소 및 시각화
- 인터랙티브한 GUI로 세그먼트 선택 및 재생
- 선택된 세그먼트의 스펙트로그램 시각화

## 기술 스택

- Python 3.x
- 데이터 처리: NumPy, librosa
- 딥러닝: PyTorch, torchvision (ResNet18)
- 차원 축소: scikit-learn (TSNE)
- 시각화: Matplotlib
- GUI: Tkinter
- 오디오 재생: Pygame
- 이미지 처리: PIL

## 작동 원리

1. MP3 파일을 로드하고 64개의 동일한 길이의 세그먼트로 분할
2. 각 세그먼트마다 스펙트로그램 생성
3. 스펙트로그램 이미지를 ResNet18에 입력하여 512차원 특징 벡터 추출
4. t-SNE 알고리즘을 통해 512차원 특징 벡터를 2차원으로 축소
5. 2차원 공간에 점으로 표시하여 시각화
6. 사용자가 점을 클릭하면 해당 세그먼트의 스펙트로그램을 표시하고 오디오 재생

## 사용 방법

1. 코드 내의 `mp3_path` 변수를 분석하려는 MP3 파일 경로로 수정
2. 스크립트 실행
3. 왼쪽 t-SNE 플롯에서 점을 클릭하여 해당 오디오 세그먼트 탐색
4. 오른쪽에 해당 세그먼트의 스펙트로그램이 표시되고 오디오가 재생됨

## 설치 요구사항

```
pip install numpy matplotlib librosa scikit-learn pygame pysoundfile torch torchvision pillow parselmouth
```

## 참고사항

- Mac에서 실행 시 GPU 가속을 위해 `mps` 장치를 사용합니다.
- 다른 환경에서 실행할 경우 `device = torch.device("mps")` 부분을 수정해야 할 수 있습니다.
  - CUDA: `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  - CPU 전용: `device = torch.device("cpu")`

## 응용 분야

- 오디오 신호 분석 및 패턴 탐색
- 음악 구조 분석
- 오디오 유사성 시각화
- 음향 데이터 클러스터링