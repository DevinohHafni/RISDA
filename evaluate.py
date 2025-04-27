import torch
from model import AlzheimerNet
from data_loader import FundusDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

def evaluate_model(test_dir, model_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_dataset = FundusDataset(test_dir, transform)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    model = AlzheimerNet().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    print("Classification Report:")
    print(classification_report(all_labels, all_preds, target_names=test_dataset.classes))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(all_labels, all_preds))

if __name__ == '__main__':
    evaluate_model('test_fundus', 'best_model.pth')
