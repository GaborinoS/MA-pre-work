import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.models as models

import os
import numpy as np
import librosa
import datetime
from tqdm import tqdm
import torchaudio.transforms as T
import matplotlib.pyplot as plt

import config
from utils_dir import transforms 
from DL_finetune import ESC_50_DL_finetune_ZUG as DSf

# Function Definitions
def hotEncoder(v):
    ret_vec = torch.zeros(v.shape[0], config.class_numbers).to(device)
    for s in range(v.shape[0]):
        ret_vec[s][v[s]] = 1
    return ret_vec

def cross_entropy_one_hot(input, target):
    _, labels = target.max(dim=1)
    return nn.CrossEntropyLoss()(input, labels)

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class Resnet50_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50_Classifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)

num_epochs = 1
learning_rate = 0.001
weight_decay = 1e-5
batch_size = 32
patience = 60
early_stop_counter = 0

train_loader, test_loader = DSf.create_generators_finetune()
num_classes = 4  
model = Resnet50_Classifier(num_classes=num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H')
log_dir = f"./results_standalone/ResnetClassifier-{current_date}-epochs-{num_epochs}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file_path = os.path.join(log_dir, "training_log.txt")
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for _, (file_name, data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)
        one_hot_labels = hotEncoder(labels)  # Convert to one-hot encoded labels
        outputs = model(data)
        loss = cross_entropy_one_hot(outputs, one_hot_labels)  # Using the new loss function
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
    avg_train_loss = train_loss / len(train_loader)

    # Validation loop
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for _, (file_name, data, labels) in tqdm(enumerate(test_loader), total=len(test_loader)):
            data, labels = data.to(device), labels.to(device)
            one_hot_labels = hotEncoder(labels)  # Convert to one-hot encoded labels
            outputs = model(data)
            loss = cross_entropy_one_hot(outputs, one_hot_labels)  # Using the new loss function
            val_loss += loss.item()

            
    avg_val_loss = val_loss / len(test_loader)
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        print("Validation Loss improved! Saving the model...")
        torch.save(model, log_dir + '/checkpoint.pth')
        early_stop_counter = 0
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            print("Early stopping!")
            break

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")

from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
train_loader, test_loader = DSf.create_generators_finetune()
model = torch.load('results_standalone/ResnetClassifier-2023-09-26-09-epochs-400/checkpoint.pth')

def evaluate_model_standalone(test_loader, model):
    model.eval()
    true_labels = []
    pred_labels = []
    with torch.no_grad():
        for (file_name, data, labels) in tqdm(test_loader):
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())

    correct_preds = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct_preds / len(true_labels)
    precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    conf_mat = confusion_matrix(true_labels, pred_labels)
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_mat)

evaluate_model_standalone(test_loader, model)