import torch
import torch.nn as nn
import torch.nn.functional as F
from efficientnet_pytorch import EfficientNet

class AlzheimerNet(nn.Module):
    def __init__(self, num_classes=4):
        super(AlzheimerNet, self).__init__()
        self.base_model = EfficientNet.from_pretrained('efficientnet-b5')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(1000, num_classes)
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def initialize_model():
    model = AlzheimerNet()
    return model
