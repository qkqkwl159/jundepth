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
import torchvision.models as models

class BasicDepthModel(nn.Module):
    def __init__(self):
        super(BasicDepthModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.up1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.out_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)

    # def forward(self, x):
    #     x = F.relu(self.bn1(self.conv1(x)))
    #     x = self.pool(x)
    #     x = F.relu(self.bn2(self.conv2(x)))
    #     x = F.relu(self.bn3(self.conv3(x)))
    #     x = F.relu(self.up1(x))
    #     x = F.relu(self.up2(x))
    #     x = self.out_conv(x)
    #     return x
    def forward(self, x):
        print("Initial size:", x.size())
        x = F.relu(self.bn1(self.conv1(x)))
        print("After conv1:", x.size())
        x = self.pool(x)
        print("After pool:", x.size())
        x = F.relu(self.bn2(self.conv2(x)))
        print("After conv2:", x.size())
        x = F.relu(self.bn3(self.conv3(x)))
        print("After conv3:", x.size())
        x = F.relu(self.up1(x))
        print("After up1:", x.size())
        x = F.relu(self.up2(x))
        print("After up2:", x.size())
        x = self.out_conv(x)
        print("Final output size:", x.size())
        x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)  # 크기를 원래 입력 크기에 맞춤
        return x

# ResNet50을 사용한 깊이 추정 모델
# class ResNetDepthModel(nn.Module):
#     def __init__(self):
#         super(ResNetDepthModel, self).__init__()
#         # 사전 훈련된 ResNet50 불러오기
#         resnet = models.resnet50(pretrained=True)
        
#         # ResNet50의 마지막 FC 레이어를 제외한 모든 레이어를 사용
#         self.features = nn.Sequential(*list(resnet.children())[:-2])
        
#         # 디코더 레이어
#         self.up1 = nn.ConvTranspose2d(2048, 1024, kernel_size=4, stride=2, padding=1)
#         self.bn1 = nn.BatchNorm2d(1024)
#         self.up2 = nn.ConvTranspose2d(1024, 512, kernel_size=4, stride=2, padding=1)
#         self.bn2 = nn.BatchNorm2d(512)
#         self.up3 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
#         self.bn3 = nn.BatchNorm2d(256)
#         self.up4 = nn.ConvTranspose2d(256, 64, kernel_size=4, stride=2, padding=1)
#         self.bn4 = nn.BatchNorm2d(64)
#         self.out_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1)
        
#     def forward(self, x):
#         x = self.features(x)
#         x = F.relu(self.bn1(self.up1(x)))
#         x = F.relu(self.bn2(self.up2(x)))
#         x = F.relu(self.bn3(self.up3(x)))
#         x = F.relu(self.bn4(self.up4(x)))
#         x = self.out_conv(x)
#         x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)
#         return x

from torchvision.models.resnet import ResNet18_Weights

# 사전 훈련된 ResNet18 모델을 불러옵니다.
# resnet18을 사용한 깊이 추정 모델
class ResNetDepthModel(nn.Module):
    def __init__(self):
        super(ResNetDepthModel, self).__init__()
        # 사전 훈련된 ResNet18 불러오기
        weights = ResNet18_Weights.IMAGENET1K_V1
        resnet = models.resnet18(weights=weights)
        # resnet = models.resnet18(pretrained=True)
        
        # ResNet18의 마지막 FC 레이어를 제외한 모든 레이어를 사용
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        
        # 디코더 레이어
        self.up1 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(256)
        self.up2 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.up3 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.up4 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(32)
        self.out_conv = nn.Conv2d(32, 1, kernel_size=3, stride=1, padding=1)
        
    def forward(self, x):
        x = self.features(x)
        x = F.relu(self.bn1(self.up1(x)))
        x = F.relu(self.bn2(self.up2(x)))
        x = F.relu(self.bn3(self.up3(x)))
        x = F.relu(self.bn4(self.up4(x)))
        x = self.out_conv(x)
        x = F.interpolate(x, size=(480, 640), mode='bilinear', align_corners=False)
        return x

#%%

if __name__ == '__main__':
        # use GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Using device:", device)
    print( torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU')

        # 데이터셋과 데이터로더 생성
    image_transform = transforms.Compose([
        transforms.Resize((240, 320)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    ])

    # NYU 데이터셋을 사용
    root_dir = "C:/Users/jun1315/Desktop/re/data/nyu/nyu"
    dataset = CustomDepthDataset(root_dir, transform=image_transform)

    # 전체 사이즈 
    total_size = len(dataset)
    # train size 70% validation size 15% test size 15%
    train_size = int(total_size * 0.7)
    validation_size = int(total_size * 0.15)
    test_size = total_size - train_size - validation_size  # 남은 데이터는 테스트 데이터로 사용

    train_dataset, validation_dataset, test_dataset = random_split(dataset, [train_size, validation_size, test_size])


    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
    validation_loader = DataLoader(validation_dataset, batch_size=64, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False, num_workers=4)


    # model = BasicDepthModel().to(device)
    model = ResNetDepthModel().to(device)
    criterion = nn.MSELoss()  # 깊이 맵은 보통 MSE로 손실 계산
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)  # 보통 Adam 사용





    num_epochs = 50  # 에포크 수
    best_val_loss = float('inf')  # 최고의 검증 손실을 저장하기 위한 변수

    # 학습 및 검증 루프
    for epoch in range(num_epochs):
        model.train()  # 모델을 학습 모드로 설정
        train_loss = 0.0
        loop = tqdm(train_loader, leave=True, desc=f'Training Epoch {epoch+1}/{num_epochs}')
        for images, depths in loop:
            images, depths = images.to(device), depths.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, depths)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * images.size(0)
            loop.set_postfix(loss=loss.item())

        train_loss = train_loss / len(train_loader.dataset)

        # 검증 단계
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0.0
        val_loop = tqdm(validation_loader, leave=True, desc=f'Validation Epoch {epoch+1}/{num_epochs}')
        with torch.no_grad():
            for images, depths in val_loop:
                images, depths = images.to(device), depths.to(device)
                outputs = model(images)
                loss = criterion(outputs, depths)
                val_loss += loss.item() * images.size(0)
                val_loop.set_postfix(val_loss=loss.item())

        val_loss = val_loss / len(validation_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'renet18_depth_model_batch4_ep50.pth')
        print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')   

