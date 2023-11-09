import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils import data
import torchvision
import torchvision.models as models


import os
import numpy as np
import librosa
import os
import torch
import torchaudio.transforms as T
import datetime
from tqdm import tqdm
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

#import own modules
import config
from utils_dir.utils import *


#empty cache
torch.cuda.empty_cache()
#os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

###############Dataloader for training the model####################
from DL_finetune import ESC_50_DL_finetune_ZUG as DSf

class Resnet50_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50_Classifier, self).__init__()

        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)

        # Change the first layer to accept 1-channel input (instead of the default 3 channels for RGB)
        self.resnet50.conv1 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Modify the last fully connected layer to match the number of classes
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)


# Hyperparameters
num_epochs = 400
learning_rate = 0.001
weight_decay = 1e-5  # L2 regularization
batch_size = 32

# Device configuration
#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early stopping parameters
patience = 60  # This value can be changed based on how many epochs of no improvement you're willing to wait
early_stop_counter = 0

# Initialize dataset and dataloaders
train_loader, test_loader = DSf.create_generators_finetune()


# Hyperparameters
num_epochs = 400
learning_rate = 0.001
weight_decay = 1e-5  # L2 regularization
batch_size = 32

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Early stopping parameters
patience = 60  # This value can be changed based on how many epochs of no improvement you're willing to wait
early_stop_counter = 0

# Initialize dataset and dataloaders
train_loader, test_loader = DSf.create_generators_finetune()

# Initialize the Classifier
num_classes = 4  
model = Resnet50_Classifier(num_classes=num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Assuming the WarmUpExponentialLR class and config values are available
scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= 100, gamma=0.95)  # Adjust warm_epochs and gamma as needed

# Create log directory
current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H')
log_dir = f"./results_standalone/ResnetClassifier-{current_date}-epochs-{num_epochs}"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

# Log file path
log_file_path = os.path.join(log_dir, "training_log.txt")

# Variables for checkpointing
best_val_loss = float('inf')

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    
    for _, (file_name, data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        data, labels = data.to(device), labels.to(device)

        # Forward pass
        outputs = model(data)
        loss = criterion(outputs, labels)

        # Backward pass and optimization
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
            outputs = model(data)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
    avg_val_loss = val_loss / len(test_loader)

    # Check for early stopping
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

    scheduler.step()  # update the learning rate

    print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
    # Log to file
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")

# Import necessary libraries
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Initialize dataset and dataloaders
train_loader, test_loader = DSf.create_generators_finetune()

model = torch.load('results_standalone/ResnetClassifier-2023-10-04-17-epochs-400/checkpoint.pth')


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
            
    # Calculate accuracy
    correct_preds = sum(t == p for t, p in zip(true_labels, pred_labels))
    accuracy = correct_preds / len(true_labels)

    # Calculate precision, recall, F1-score
    precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')

    # Calculate the confusion matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)

    # Print the results
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score * 100:.2f}%")
    print("Confusion Matrix:")
    print(conf_mat)

# Call the evaluate function after training
evaluate_model_standalone(test_loader, model)
