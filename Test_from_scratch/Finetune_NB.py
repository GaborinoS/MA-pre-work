if __name__ == '__main__':
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils import data
    import torchvision
    
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
    from torch import optim, nn
    
    #own modules
    import config
    from utils_dir.utils import *
    
    #empty cache
    torch.cuda.empty_cache()
    
    
    # CUDA for PyTorch
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    
    ###############Dataloader for training the model####################
    if config.ADSMI:
        from DL_finetune import ADSMI_DL_finetune as DSf
        Data_name = 'ADSMI'
        print('ADSMI')
    if config.ESC_50:
        from DL_finetune import ESC_50_DL_finetune as DSf
        Data_name = 'ESC-50'
        print('ESC-50')
    
    class ContrastiveTripletModel(nn.Module):
        def __init__(self, embedding_dim=2048, projection_dim=128,input_channels=config.channels):
            super(ContrastiveTripletModel, self).__init__()
            self.resnet50 = models.resnet50(pretrained=False)
            
            # Modifications for your dataset:
            # Assuming your data is a spectrogram of shape [128, X]. 
            # ResNet50 expects 3-channel inputs, so let's adapt the first layer.
            self.resnet50.conv1 = nn.Conv2d(input_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            
            # Remove last FC layer to get embeddings
            self.encoder = nn.Sequential(*list(self.resnet50.children())[:-1])
            
            # Projection head
            self.projection = nn.Sequential(
                nn.Linear(embedding_dim, embedding_dim),  # 1st projection layer, can be modified
                nn.ReLU(),
                nn.Linear(embedding_dim, projection_dim)  # 2nd projection layer
            )
            
            # Dropout layer (with 50% probability, adjust as needed)
            #self.dropout = nn.Dropout(p=0.5)
    
        def forward_one(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)  # Flatten for easier downstream processing
            x = self.projection(x)  # Pass through the projection head
            #x = self.dropout(x)
            return x
    
        def forward(self, input1, input2, input3):
            output1 = self.forward_one(input1)
            output2 = self.forward_one(input2)
            output3 = self.forward_one(input3)
            return output1, output2, output3
    
    log_dir = "./results/CLR-2035-10-17-Cluster_TRY1"
    
    
    class FineTuneModel(nn.Module):
        def __init__(self, encoder, num_classes):
            super(FineTuneModel, self).__init__()
            self.encoder = encoder
            # Insert input size of the encoder here 2048
            self.classifier = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                #nn.Dropout(0.5),
                nn.Linear(512, num_classes)
            )
            
        def forward(self, x):
            x = self.encoder(x)
            x = x.view(x.size(0), -1)  # Flatten the output
            x = self.classifier(x)
            return x
    
    
    
    # Hyperparameters
    num_epochs = config.finetune_epochs
    learning_rate = config.lr
    weight_decay = 1e-5  # L2 regularization
    batch_size = config.batch_size
    num_classes = config.class_numbers  # Adjust this to the number of classes in your dataset
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Early stopping parameters
    patience = config.patience  # This value can be changed based on how many epochs of no improvement you're willing to wait
    early_stop_counter = 0
    
    # Initialize dataset and dataloaders
    train_loader, test_loader = DSf.create_generators_finetune()
    
    # Load the entire pre-trained model (from your contrastive training)
    pretrained_model = torch.load(log_dir + '/checkpoint.pth')
    encoder_trained = pretrained_model.encoder
    
    # Initialize the FineTuneModel with the pre-trained encoder
    model = FineTuneModel(encoder_trained, num_classes).to(device)
    
    # optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Assuming the WarmUpExponentialLR class and config values are available
    scheduler = WarmUpExponentialLR(optimizer, cold_epochs= 0, warm_epochs= config.warm_epochs, gamma=config.gamma)  # Adjust warm_epochs and gamma as needed
    
    
    # One-hot encoding and custom loss function
    def hotEncoder(v):
        ret_vec = torch.zeros(v.shape[0], num_classes).to(device)
        for s in range(v.shape[0]):
            ret_vec[s][v[s]] = 1
        return ret_vec
    
    def cross_entropy_one_hot(input, target):
        _, labels = target.max(dim=1)
        return nn.CrossEntropyLoss()(input, labels)
    
    # Create log directory
    current_date = datetime.datetime.now().strftime('%Y-%m-%d-%H')
    log_dir = f"./finetune_results/FineTune-{current_date}-epochs-{num_epochs}-{Data_name}"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Log file path
    log_file_path = os.path.join(log_dir, "training_log.txt")
    
    #write in log file the hyperparameters
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"##############################################################################\n")
        log_file.write(f"Hyperparameters:\n")
        if config.ADSMI:
            log_file.write(f"ADSMI labeled Data\n")
        if config.ESC_50:
            log_file.write(f"ESC-50 labeled Data\n")
    
        log_file.write(f"num_epochs: {num_epochs}\n")
        log_file.write(f"initial_learning_rate: {learning_rate}\n")
        log_file.write(f"weight_decay: {weight_decay}\n")
        log_file.write(f"batch_size: {batch_size}\n")
        log_file.write(f"patience: {patience}\n")
        log_file.write(f"early_stop_counter: {early_stop_counter}\n")
        log_file.write(f"num_classes: {num_classes}\n")
        #log_file.write(f"model: {model}\n")
        log_file.write(f"criterion: CrossEntropyLoss()\n")
        #log_file.write(f"optimizer: {optimizer}\n")
        #log_file.write(f"scheduler: {scheduler}\n")
        #log_file.write(f"log_dir: {log_dir}\n")
        log_file.write(f"log_file_path: {log_file_path}\n")
        #log_file.write(f"train_loader: {train_loader}\n")
        #log_file.write(f"test_loader: {test_loader}\n")
        log_file.write(f"##############################################################################\n")
    
    # Variables for checkpointing
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
    
        for _, (file_name, data, labels) in tqdm(enumerate(train_loader), total=len(train_loader)):
            data = data.to(device)
            labels = labels.to(device).unsqueeze(1)
            label_vec = hotEncoder(labels)
    
            # Forward pass
            outputs = model(data)
            loss = cross_entropy_one_hot(outputs, label_vec)
    
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
                data = data.to(device)
                labels = labels.to(device).unsqueeze(1)
                label_vec = hotEncoder(labels)
                outputs = model(data)
                loss = cross_entropy_one_hot(outputs, label_vec)
                val_loss += loss.item()
    
        avg_val_loss = val_loss / len(test_loader)
    
        # Check for early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print("Validation Loss improved! Saving the model...")
    
            # Log to file
            with open(log_file_path, 'a') as log_file:
                log_file.write(f"Validation Loss improved! Saving the model...\n")
            
            torch.save(model, log_dir + '/checkpoint.pth')
            early_stop_counter = 0
        else:
            early_stop_counter += 1
            if early_stop_counter >= patience:
                print("Early stopping!")
                break
        
        #scheduler step update the learning rate  
        scheduler.step()         
    
        print(f'Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
    
        # Log to file early stoping counter
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Early stopping counter: {early_stop_counter} from {patience}\n")
    
        # Log to file
        with open(log_file_path, 'a') as log_file:
            log_file.write(f"Epoch [{epoch+1}/{num_epochs}] Training Loss: {avg_train_loss:.4f}, Validation Loss: {avg_val_loss:.4f}\n")
            log_file.write(f"##\n")
    
    
    from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
    import numpy as np
    
    # Initialize variables to store the true and predicted labels
    true_labels = []
    pred_labels = []
    
    # Set model to evaluation mode
    model = torch.load('finetune_results/FineTune-2023-09-30-17-epochs-400/checkpoint.pth')
    model.eval()
    
    # Initialize dataset and dataloaders
    train_loader, test_loader = DSf.create_generators_finetune()
    
    # Evaluate the model on the test dataset
    with torch.no_grad():
        for (file_name, data, labels) in tqdm(test_loader):
            data = data.to(device)
            labels = labels.to(device)
            
            outputs = model(data)
            
            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(torch.argmax(outputs, dim=1).cpu().numpy())
    
    # Convert lists to arrays for better indexing and operations
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    
    # Calculate accuracy
    correct_preds = np.sum(true_labels == pred_labels)
    accuracy = correct_preds / len(true_labels)
    
    # Calculate precision, recall, F1-score, and support
    precision, recall, f1_score, support = precision_recall_fscore_support(true_labels, pred_labels, average='weighted')
    
    # Calculate the confusion matrix
    conf_mat = confusion_matrix(true_labels, pred_labels)
    
    # Print the classification report
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score * 100:.2f}%")
    
    # Print the confusion matrix
    print("Confusion Matrix:")
    print(conf_mat)
    