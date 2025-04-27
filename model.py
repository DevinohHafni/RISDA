import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return loss.mean()

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b3')
        self.vessel_branch = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2))
        self.fc = nn.Sequential(
            nn.Linear(1000 + 32*128*128, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes))
    
    def forward(self, x):
        img_features = self.base_model(x)
        vessel_features = self.vessel_branch(x)
        vessel_features = vessel_features.view(vessel_features.size(0), -1)
        combined = torch.cat([img_features, vessel_features], dim=1)
        return self.fc(combined)

def initialize_model():
    model = AlzheimerNet()
    return model
