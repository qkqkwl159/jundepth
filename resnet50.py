#%%
#%%
import torch
import torch.nn as nn
from torchvision import models, transforms
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import os
import glob
#%%
# Load the dataset

#%%
# Set device to GPU if available, else CPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)
print( torch.cuda.get_device_name() if device.type == 'cuda' else 'CPU')

#%%
# Custom Dataset class for NYU Depth V2
# class NYUDepthV2Dataset(Dataset):
#     def __init__(self, data, transform=None):
#         self.data = data
#         self.transform = transform

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         image_data = self.data[idx]["image"]
#         depth_data = self.data[idx]["depth_map"]
#         image = Image.fromarray(image_data)
#         depth = Image.fromarray(depth_data).unsqueeze(0).float()

#         if self.transform:
#             image = self.transform(image)
#         return image, depth

# # Define transformations
# image_transform = transforms.Compose([
#     transforms.Resize((240, 320)),  # Resize to the input size expected by the network
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalization parameters
# ])

# # Load training data
# train_set = ds["train"]
# nyu_depth_dataset = NYUDepthV2Dataset(train_set, transform=image_transform)
# train_loader = DataLoader(nyu_depth_dataset, batch_size=4, shuffle=True, num_workers=4)

class CustomDepthDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(root_dir, '**/*.jpg'), recursive=True))
        self.depth_paths = sorted(glob.glob(os.path.join(root_dir, '**/*.png'), recursive=True))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        depth = Image.open(self.depth_paths[idx]).convert('L')

        if self.transform:
            image = self.transform(image)

        depth = transforms.ToTensor()(depth).float()

        return image, depth
    
image_transform = transforms.Compose([
    transforms.Resize((240, 320)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


root_dir = "C:/Users/jun1315/Desktop/re/data/nyu/nyu"
dataset = CustomDepthDataset(root_dir, transform=image_transform)
data_loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=4)
#%%

# Model definition
class DepthEstimationModel(nn.Module):
    def __init__(self):
        super(DepthEstimationModel, self).__init__()
        resnet = models.resnet50(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-2])
        self.decoder = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 1, kernel_size=3, padding=1)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.decoder(x)
        return x

# Initialize model, loss, and optimizer
model = DepthEstimationModel().to(device)
criterion = nn.L1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
#%%

# Function to visualize data samples
def visualize_batch(images, depths):
    fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(12, 6))
    for i in range(min(4, images.shape[0])):  # To handle cases where the batch size is < 4
        img = images[i].cpu().detach().numpy().transpose((1, 2, 0))
        img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])  # Unnormalize
        img = np.clip(img, 0, 1)
        depth = depths[i].cpu().detach().numpy().squeeze()

        axs[0, i].imshow(img)
        axs[0, i].axis('off')
        axs[1, i].imshow(depth, cmap='viridis')
        axs[1, i].axis('off')
    plt.show()
#%%
num_epochs = 5

print("Training the model...")
for epoch in range(num_epochs):
    loop = tqdm(data_loader, leave=True)
    for images, true_depths in loop:
        images, true_depths = images.to(device), true_depths.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, true_depths)
        loss.backward()
        optimizer.step()

        loop.set_description(f'Epoch [{epoch+1}/{num_epochs}]')
        loop.set_postfix(loss=loss.item())
#%%

# # Visualize a batch of data
# data_iter = iter(train_loader)
# images, depths = next(data_iter)
# images, depths = images.to(device), depths.to(device)
# visualize_batch(images, depths)

# Save the trained model
torch.save(model.state_dict(), 'depth_estimation_model.pth')
