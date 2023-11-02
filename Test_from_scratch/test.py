import torch
import torch.nn.functional as F
import torch.optim as optim
import torchaudio.transforms as T
import torchvision.models as models
#import torchaudio
#import os
#from torch.utils.data import Dataset, DataLoader
#import random
import pandas as pd
import torch.nn as nn
from tqdm import tqdm
import torch.optim.lr_scheduler as lr_scheduler
# Implement Stratified K-Folds Cross-validation
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split
#import matplotlib.pyplot as plt
#import numpy as np

import config



if config.ADSMI:
    from DL_finetune import ADSMI_DL_TVsplit as DL

print(config.channels)

class Resnet50_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(Resnet50_Classifier, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.conv1 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet50.fc.in_features
        self.resnet50.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet50(x)
    



class ResNet101_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ResNet101_Classifier, self).__init__()
        self.resnet101 = models.resnet101(pretrained=True)
        # Modify the input layer to match your input data channels
        # If your input data has different channels, adjust this line accordingly
        self.resnet101.conv1 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        num_features = self.resnet101.fc.in_features
        self.resnet101.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.resnet101(x)
    

    
class ModifiedResnet50_Classifier(nn.Module):
    def __init__(self, num_classes):
        super(ModifiedResnet50_Classifier, self).__init__()
        
        # Load the pretrained ResNet-50 model
        self.resnet50 = models.resnet50(pretrained=True)
        
        # Modify the first convolutional layer
        self.resnet50.conv1 = nn.Conv2d(config.channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        
        # Remove the final fully connected layer
        self.resnet50.fc = nn.Identity()  # Set the final layer to an identity mapping
        
        # Define the custom fully connected layers
        num_features = 2048 
        self.fc1 = nn.Linear(num_features, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.resnet50(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
    
#------Datasplit
# Load the dataframe
labels_file = pd.read_csv('./data/labeled_ADSMI/labels_int.csv', index_col=0)
train_df, temp = train_test_split(labels_file, test_size=0.2, stratify=labels_file['Label_int'], random_state=55)
test_df, val_df = train_test_split(temp, test_size=0.5, stratify=temp['Label_int'], random_state=55)
# train test split
print("Train size: ", len(train_df))
print("Test size: ", len(test_df))
print("Val size: ", len(val_df))




#------Data fold generation for cross-validation
n_folds = 8
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

train_loader, test_loader = DL.create_generators_finetune_train(train_df,test_df)

#  Create an instance of the model
num_classes = len(set(labels_file["Label_int"]))  # Assuming the number of classes is the unique count of "Label_int" in your labels_file
model = Resnet50_Classifier(num_classes)
#model = ModifiedResnet50_Classifier(num_classes)
#model = ResNet101_Classifier(num_classes)



#  Transfer the model to the appropriate device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Define a loss function and optimizer
criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004, weight_decay = 1e-4 ) # Adjust the value as needed)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5, verbose=True)

train_losses = []
test_losses = []
config.best_accuracy = 0
config.model_path = "./results_standalone/test_re1Layer_checkpoint.pth"
# Training loop
num_epochs = 1  # Adjust this as needed
for epoch in range(num_epochs):
    model.train()
    for batch_idx, (spectrograms, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
        spectrograms = spectrograms.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(spectrograms)
        loss = criterion(outputs, labels)
        train_losses.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



    correct_predictions = 0
    total_samples = 0

    # Set the model to evaluation mode (important for dropout and batch normalization)
    model.eval()

    # Iterate through the test set
    with torch.no_grad():  # Disable gradient computation during testing
        for spectrograms, labels in test_loader:
            # Move data to the testing device
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            loss = criterion(outputs, labels)
            test_losses.append(loss.item())
            
            # Compute the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Update evaluation metrics
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()

    #if new test accuracy is better than the previous best, save the model
    if correct_predictions / total_samples > config.best_accuracy:
        config.best_accuracy = correct_predictions / total_samples
        torch.save(model, config.model_path)
        
    # Step the learning rate scheduler
    scheduler.step(test_losses[-1])

    # Calculate accuracy or other evaluation metrics
    accuracy = correct_predictions / total_samples
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}")
    print(f"Test Accuracy: {accuracy * 100:.2f}%")

