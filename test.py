#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import numpy as np
import os
import glob
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
import matplotlib.pyplot as plt
from basicDataloader import CustomDepthDataset
from tqdm import tqdm
from basicCNN import BasicDepthModel
from basicCNN import ResNetDepthModel


def plot_depth_maps(images, true_depths, predicted_depths, num_images=3):
    plt.figure(figsize=(10, num_images * 3))
    
    for i in range(num_images):
        # 원본 이미지
        plt.subplot(num_images, 3, i * 3 + 1)
        plt.imshow(images[i].permute(1, 2, 0).cpu().numpy())
        plt.title('Original Image')
        plt.axis('off')

        # 실제 깊이 맵
        plt.subplot(num_images, 3, i * 3 + 2)
        plt.imshow(true_depths[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title('True Depth Map')
        plt.axis('off')

        # 예측 깊이 맵
        plt.subplot(num_images, 3, i * 3 + 3)
        plt.imshow(predicted_depths[i].squeeze().cpu().numpy(), cmap='gray')
        plt.title('Predicted Depth Map')
        plt.axis('off')
    
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print( torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')
    image_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    # NYU 데이터셋을 사용
    root_dir = "C:/Users/jun1315/Desktop/re/data/nyu/nyu"
    dataset = CustomDepthDataset(root_dir, transform=image_transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # 전체 사이즈 
    total_size = len(dataset)
    # train size 70% validation size 15% test size 15%
    train_size = int(total_size * 0.7)
    validation_size = int(total_size * 0.15)
    test_size = total_size - train_size - validation_size  # 남은 데이터는 테스트 데이터로 사용

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)

    # model = BasicDepthModel().to(device)
    model = ResNetDepthModel().to(device)
    criterion = nn.MSELoss()  # 깊이 맵은 보통 MSE로 손실 계산
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 보통 Adam 사용


    # 테스트 단계
    # 모델을 테스트 모드로 설정
    model.eval()
    images, true_depths = next(iter(test_loader))  # 테스트 로더에서 샘플 배치 가져오기
    test_loss = 0.0
    test_loop = tqdm(test_loader, leave=True, desc='./resnet18_depth_model_batch4_ep50.pth')
    with torch.no_grad():
        for images, depths in test_loop:
            images, depths = images.to(device), depths.to(device)
            predicted_depths = model(images)
            loss = criterion(predicted_depths, depths)
            test_loss += loss.item() * images.size(0)
            test_loop.set_postfix(test_loss=loss.item())

    test_loss = test_loss / len(test_loader.dataset)
    print(f'Test Loss: {test_loss:.4f}')

    # 결과 시각화
    plot_depth_maps(images, true_depths, predicted_depths, num_images=3)
