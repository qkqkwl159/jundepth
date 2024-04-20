import torch
import cv2
import urllib.request
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# 이미지 다운로드
url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
urllib.request.urlretrieve(url, filename)

# MiDaS 모델 로드
model_type = "DPT_Large"
midas = torch.hub.load("intel-isl/MiDaS", model_type)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas.to(device)
midas.eval()

# 변환 로드 및 선택
midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# 이미지 읽기 및 변환
img = cv2.imread(filename)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
input_tensor = transform(img).to(device)

# 기준 입력 설정
baseline = torch.zeros_like(input_tensor).to(device)
input_tensor.requires_grad_(True)

# Integrated Gradients 계산
steps = 50  # 적분을 위한 스텝 수
integrated_gradients = torch.zeros_like(input_tensor)
for alpha in np.linspace(0, 1, steps):
    interpolated_input = baseline + alpha * (input_tensor - baseline)
    interpolated_input.requires_grad_(True)
    pred = midas(interpolated_input)
    pred.backward(torch.ones_like(pred))
    integrated_gradients += interpolated_input.grad
    interpolated_input.grad.zero_()
integrated_gradients /= steps

# Attribution 맵 계산
attributions = (input_tensor - baseline) * integrated_gradients

# 깊이 맵 생성
with torch.no_grad():
    depth_map = midas(input_tensor)

# 깊이 맵 재조정
depth_map = torch.nn.functional.interpolate(
    depth_map.unsqueeze(1),
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

# 시각화
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# 원본 이미지
axes[0].imshow(img)
axes[0].set_title('Original Image')
axes[0].axis('off')

# 깊이 맵
depth_map_np = depth_map.cpu().numpy()
axes[1].imshow(depth_map_np, cmap='viridis')
axes[1].set_title('Depth Map')
axes[1].axis('off')

# Attribution Map
attributions_np = attributions.cpu().detach().numpy().squeeze()
# 평균을 취하는 대신에 채널 차원을 함께 평면화하여 모든 변화량을 합산합니다.
attributions_np = np.sum(np.abs(attributions_np), axis=0)
axes[2].imshow(attributions_np, cmap='hot')
axes[2].set_title('Attribution Map')
axes[2].axis('off')

plt.tight_layout()
plt.show()
