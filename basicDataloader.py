import os
import glob
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

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