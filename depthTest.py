# import cv2
# import torch
# import urllib.request

# import matplotlib.pyplot as plt

# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# model_type = "DPT_Large"     # MiDaS v3 - Large     (highest accuracy, slowest inference speed)
# #model_type = "DPT_Hybrid"   # MiDaS v3 - Hybrid    (medium accuracy, medium inference speed)
# #model_type = "MiDaS_small"  # MiDaS v2.1 - Small   (lowest accuracy, highest inference speed)

# midas = torch.hub.load("intel-isl/MiDaS", model_type)


# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# midas.to(device)
# midas.eval()

# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")

# if model_type == "DPT_Large" or model_type == "DPT_Hybrid":
#     transform = midas_transforms.dpt_transform
# else:
#     transform = midas_transforms.small_transform



# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# input_batch = transform(img).to(device)


# with torch.no_grad():
#     prediction = midas(input_batch)

#     prediction = torch.nn.functional.interpolate(
#         prediction.unsqueeze(1),
#         size=img.shape[:2],
#         mode="bicubic",
#         align_corners=False,
#     ).squeeze()

# output = prediction.cpu().numpy()

# plt.imshow(output)
# plt.show()


# import cv2
# import torch
# import urllib.request
# import numpy as np
# import matplotlib.pyplot as plt

# # 이미지 다운로드
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# # MiDaS 모델 로드
# model_type = "DPT_Large"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas.to(device)
# midas.eval()

# # 변환 로드 및 선택
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# # 이미지 읽기 및 변환
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# input_batch = transform(img).to(device)

# # 입력에 대한 그래디언트 계산 가능하도록 설정
# input_batch.requires_grad_(True)

# # 깊이 예측 및 그래디언트 계산
# with torch.no_grad():
#     prediction = midas(input_batch)

# # 스칼라 값(여기서는 이미지 중앙의 깊이)에 대한 그래디언트를 계산
# prediction[:, :, prediction.size(2) // 2, prediction.size(3) // 2].backward(torch.ones_like(prediction))

# # 그래디언트 절대값 취하기
# saliency, _ = torch.max(input_batch.grad.data.abs().squeeze(), dim=0)

# # 깊이 맵 재조정 및 시각화
# prediction = torch.nn.functional.interpolate(
#     prediction.unsqueeze(1),
#     size=img.shape[:2],
#     mode="bicubic",
#     align_corners=False,
# ).squeeze()

# depth_map = prediction.cpu().numpy()
# saliency_map = saliency.cpu().numpy()

# # 깊이 맵과 Saliency Map 시각화
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(depth_map, cmap='viridis')
# axes[0].set_title('Depth Map')
# axes[0].axis('off')

# axes[1].imshow(saliency_map, cmap='hot')
# axes[1].set_title('Saliency Map')
# axes[1].axis('off')

# plt.show()



# import cv2
# import torch
# import urllib.request
# import numpy as np
# import matplotlib.pyplot as plt

# # 이미지 다운로드
# url, filename = ("https://github.com/pytorch/hub/raw/master/images/dog.jpg", "dog.jpg")
# urllib.request.urlretrieve(url, filename)

# # MiDaS 모델 로드
# model_type = "DPT_Large"
# midas = torch.hub.load("intel-isl/MiDaS", model_type)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# midas.to(device)
# midas.eval()

# # 변환 로드 및 선택
# midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
# transform = midas_transforms.dpt_transform if model_type in ["DPT_Large", "DPT_Hybrid"] else midas_transforms.small_transform

# # 이미지 읽기 및 변환
# img = cv2.imread(filename)
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# input_batch = transform(img).to(device)

# # 입력에 대한 그래디언트 계산 가능하도록 설정
# input_batch.requires_grad_(True)

# # 깊이 예측 및 그래디언트 계산
# prediction = midas(input_batch)

# # Saliency map 계산을 위해 사용될 예측 중 하나 선택
# prediction = prediction.unsqueeze(1)  # 필요 시 차원 추가
# target = prediction[:, 0, :, :]  # 채널 차원이 있어야 합니다.
# target = target.mean()  # 전체 평균을 사용하여 스칼라 값을 생성
# target.backward()  # 스칼라 값에 대해 backward 실행

# # 그래디언트 절대값 취하기
# saliency, _ = torch.max(input_batch.grad.data.abs().squeeze(), dim=0)

# # 깊이 맵 재조정 및 시각화
# prediction = torch.nn.functional.interpolate(
#     prediction,
#     size=img.shape[:2],
#     mode="bicubic",
#     align_corners=False,
# ).squeeze()

# depth_map = prediction.cpu().numpy()
# saliency_map = saliency.cpu().numpy()

# # 깊이 맵과 Saliency Map 시각화
# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(depth_map, cmap='viridis')
# axes[0].set_title('Depth Map')
# axes[0].axis('off')

# axes[1].imshow(saliency_map, cmap='hot')
# axes[1].set_title('Saliency Map')
# axes[1].axis('off')

# plt.show()


import cv2
import torch
import urllib.request
import numpy as np
import matplotlib.pyplot as plt

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
input_batch = transform(img).to(device)

# 입력에 대한 그래디언트 계산 가능하도록 설정
input_batch.requires_grad_(True)

# 깊이 예측 및 그래디언트 계산
prediction = midas(input_batch)

# Saliency map 계산을 위해 사용될 예측 중 하나 선택
prediction = prediction.unsqueeze(1)  # 필요 시 차원 추가
target = prediction.mean()  # 전체 평균을 사용하여 스칼라 값을 생성
target.backward()  # 스칼라 값에 대해 backward 실행

# 그래디언트 절대값 취하기
saliency, _ = torch.max(input_batch.grad.data.abs().squeeze(), dim=0)

# 깊이 맵 재조정 및 시각화
prediction = torch.nn.functional.interpolate(
    prediction,
    size=img.shape[:2],
    mode="bicubic",
    align_corners=False,
).squeeze()

# .detach() 메소드를 사용하여 그래디언트 추적 제거
depth_map = prediction.cpu().detach().numpy()
saliency_map = saliency.cpu().detach().numpy()

# 깊이 맵과 Saliency Map 시각화
fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(depth_map, cmap='viridis')
axes[0].set_title('Depth Map')
axes[0].axis('off')

axes[1].imshow(saliency_map, cmap='hot')
axes[1].set_title('Saliency Map')
axes[1].axis('off')

plt.show()
