import tkinter as tk
import matplotlib.pyplot as plt
import matplotlib.backends.backend_tkagg as tkagg
import parselmouth
import numpy as np
from sklearn.manifold import TSNE
import librosa
from pygame import mixer
import soundfile as sf
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import io

# MP3 파일 경로
mp3_path = '/Users/air/Desktop/r_ParaElisa_44.mp3'
mp3_filename = mp3_path.split('/')[-1]  # 파일명 추출

# ResNet18 모델 로드 및 설정
device = torch.device("mps")
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])  # 마지막 분류층 제거
model.to(device)
model.eval()

# 이미지 전처리를 위한 transform 정의
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 오디오 파일 로드
y, sr = librosa.load(mp3_path)

# 오디오를 64개의 섹션으로 분할
slices = np.array_split(y, 64)

# 각 섹션에서 특징 추출
features = []
for s in slices:
    # 스펙트로그램 생성
    D = librosa.amplitude_to_db(np.abs(librosa.stft(s)), ref=np.max)
    
    # matplotlib을 사용하여 스펙트로그램을 이미지로 변환
    plt.figure(figsize=(3, 3))
    plt.axis('off')
    plt.imshow(D, aspect='auto', origin='lower')
    
    # 플롯을 이미지로 저장
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # PIL Image로 변환
    buf.seek(0)
    image = Image.open(buf).convert('RGB')
    
    # ResNet 입력 형식으로 변환
    input_tensor = transform(image).unsqueeze(0).to(device)
    
    # 특징 추출
    with torch.no_grad():
        feature = model(input_tensor)
        feature = feature.squeeze().cpu().numpy()
    
    features.append(feature)
    buf.close()

features = np.array(features)
features = features.reshape(64, -1)  # 특징 벡터 평탄화

# t-SNE를 사용하여 데이터를 플로팅
tsne = TSNE(n_components=2)
X_2d = tsne.fit_transform(features)

# pygame 초기화
mixer.init()

def play_clip(idx):
    # 오디오 클립 재생
    slice = slices[idx]
    sf.write('temp.wav', slice, sr)  # 임시 WAV 파일로 저장
    mixer.music.load('temp.wav')  # pygame으로 로드
    mixer.music.play()  # 재생

# GUI 생성
root = tk.Tk()
root.title(mp3_filename)  # 창 제목을 MP3 파일명으로 설정

# Figure 생성 (2개의 서브플롯)
fig = plt.figure(figsize=(12, 5))
ax1 = plt.subplot(121)  # t-SNE 플롯
ax2 = plt.subplot(122)  # 스펙트로그램

# t-SNE 플롯 그리기
ax1.scatter(X_2d[:, 0], X_2d[:, 1])
ax1.set_title('t-SNE Plot')

# 초기 스펙트로그램 영역 설정
ax2.set_title('Spectrogram')
ax2.axis('off')

canvas = tkagg.FigureCanvasTkAgg(fig, master=root)
canvas.draw()
canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

# 마우스 이벤트 처리
def on_click(event):
    if event.inaxes == ax1:  # t-SNE 플롯 영역에서만 동작
        for i, (x, y) in enumerate(X_2d):
            if abs(x - event.xdata) < 0.1 and abs(y - event.ydata) < 0.1:
                # 오디오 재생
                play_clip(i)
                
                # 스펙트로그램 업데이트
                ax2.clear()
                ax2.set_title(f'Spectrogram of Segment {i+1}')
                
                # 선택된 슬라이스의 스펙트로그램 생성
                D = librosa.amplitude_to_db(np.abs(librosa.stft(slices[i])), ref=np.max)
                img = ax2.imshow(D, aspect='auto', origin='lower', cmap='viridis')
                ax2.axis('off')
                
                # 캔버스 업데이트
                canvas.draw()
                break

canvas.mpl_connect('button_press_event', on_click)

# 윈도우 크기 조정
root.geometry('1200x500')  # 적절한 윈도우 크기 설정

root.mainloop()