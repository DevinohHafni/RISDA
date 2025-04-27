import os
import random
import numpy as np
from PIL import Image, ImageOps, ImageEnhance
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import cv2

class AddPlaques:
    def __call__(self, img):
        img = np.array(img)
        h, w = img.shape[:2]
        for _ in range(random.randint(5, 10)):
            x, y = random.randint(0, w-1), random.randint(0, h-1)
            radius = random.randint(2, 5)
            cv2.circle(img, (x, y), radius, (255, 255, 200), -1)
        return Image.fromarray(img)

def add_vessel_noise(img, intensity=0.1):
    img = np.array(img)
    h, w = img.shape[:2]
    noise = np.random.uniform(-intensity, intensity, (h, w, 3))
    img = np.clip(img + (noise * 255), 0, 255).astype(np.uint8)
    return Image.fromarray(img)

class FundusDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Normal', 'Mild', 'Moderate', 'Severe']
        self.samples = self._load_samples()
        
    def _load_samples(self):
        samples = []
        for class_idx, class_name in enumerate(self.classes):
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                samples.append((img_path, class_idx))
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label

def get_data_loaders(train_dir, test_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        AddPlaques(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.RandomApply([transforms.Lambda(lambda x: add_vessel_noise(x))], p=0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.45, 0.30, 0.25], std=[0.22, 0.15, 0.14])
    ])
    
    train_dataset = FundusDataset(train_dir, transform)
    test_dataset = FundusDataset(test_dir, transform)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, test_loader
