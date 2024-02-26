import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from DL_finetune import ADSMI_DL_TVsplit as DL
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, cohen_kappa_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import config




def train_test_ADSMI():
    #------Datasplit
    # Load the dataframe
    labels_file = pd.read_csv('./data/labeled_ADSMI/labels_int.csv', index_col=0)
    #train_df, test_df = train_test_split(labels_file, test_size=0.2, stratify=labels_file['Label_int'], random_state=47)
    #val_df = test_df
    random_state = 96
    train_df, temp = train_test_split(labels_file, test_size=0.2, stratify=labels_file['Label_int'], random_state=random_state)
    test_df, val_df = train_test_split(temp, test_size=0.5, stratify=temp['Label_int'], random_state=random_state)
    # train test split
    print("Train size: ", len(train_df))
    print("Test size: ", len(test_df))
    print("Val size: ", len(val_df))

    return train_df, test_df, val_df

def validation_fun(dir_path,modelpath,measure_checkpoint,df,num_epochs,batch_size,learning_rate,weight_decay):
    import config
    # Set the device for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Create a data loader for the test set
    val_loader = DL.create_generators_finetune_val(df)  

    #model = torch.load("./results_standalone/newgpu2_checkpoint.pth")
    model = torch.load(dir_path + modelpath)
    # Transfer the model to the testing device
    model.to(device)

    # Define a criterion for evaluation (e.g., cross-entropy loss for classification)
    criterion = nn.CrossEntropyLoss()

    # Initialize variables for evaluation metrics (e.g., accuracy)
    correct_predictions = 0
    total_samples = 0

    # Define the label dictionary
    true_labels_dic = {0: '[Kreischen]', 1: '[Kreischen][Quietschen]', 2: '[Negativ]', 3: '[Quietschen]'}

    # Set the model to evaluation mode 
    model.eval()

    # Initialize lists to store all true labels and predicted labels
    all_true_labels = []
    all_predicted_labels = []

    # Iterate through the test set
    with torch.no_grad():
        for spectrograms, labels in val_loader:
            # Move data to the testing device
            spectrograms = spectrograms.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(spectrograms)
            
            # Compute the predicted labels
            _, predicted = torch.max(outputs, 1)

            # Append true and predicted labels to the lists
            all_true_labels.extend(labels.cpu().numpy())
            all_predicted_labels.extend(predicted.cpu().numpy())

    # Convert lists to numpy arrays
    all_true_labels = np.array(all_true_labels)
    all_predicted_labels = np.array(all_predicted_labels)

    # Calculate accuracy, precision, recall, F1-score, etc. using all_true_labels and all_predicted_labels
    accuracy = np.mean(all_true_labels == all_predicted_labels)
    precision, recall, f1_score, _ = precision_recall_fscore_support(all_true_labels, all_predicted_labels, average='macro')


    #calculate balanced accuracy
    balanced_accuracy2 = balanced_accuracy_score(all_true_labels, all_predicted_labels)
    #kappa score
    cohens_kappa = cohen_kappa_score(all_true_labels, all_predicted_labels)


    print(f"\nEvaluation Results:")
    #print(f"Balanced Accuracy: {balanced_accuracy * 100:.2f}%")
    print(f"Balanced Accuracy2: {balanced_accuracy2 * 100:.2f}%")

    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"\nEvaluation Results:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print(f"Precision: {precision * 100:.2f}%")
    print(f"Recall: {recall * 100:.2f}%")
    print(f"F1-score: {f1_score * 100:.2f}%")
    print(f"Kappa: {cohens_kappa * 100:.2f}%")

    conf_mat = confusion_matrix(all_true_labels, all_predicted_labels)
    def plot_confusion_matrix(conf_mat, class_labels):
        plt.figure(figsize=(7, 7))
        sns.set(font_scale=1.2)
        
        class_labels = [str(label) for label in class_labels]
        
        ax = sns.heatmap(conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False,
                    xticklabels=class_labels,
                    yticklabels=class_labels)
        
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title('Confusion Matrix')
        plt.xticks(rotation=90)

        return plt.gcf()  # Return the current figure

    import config
    #create classification results text file
    with open(f'{dir_path}/results_file_{measure_checkpoint}.txt', 'w') as log_file:
        log_file.write(f"########################################################\n")
        log_file.write(f"Frequency Mask:{config.freq_mask_param}\n")
        log_file.write(f"Time Mask: {config.time_mask_param}\n")
        log_file.write(f"Spectro:\n")
        log_file.write(f"        n_fft={2048},\n")
        log_file.write(f"        hop_length = {512},\n")
        log_file.write(f"        n_mels={128},  \n")
        log_file.write(f"        win_length = {2048},\n")
        log_file.write(f"Epochs: {num_epochs}\n")
        log_file.write(f"Batch size: {batch_size}\n")
        log_file.write(f"Optimizer: Adam\n")
        log_file.write(f"Learning rate: {learning_rate}\n")
        log_file.write(f"Weight decay: {weight_decay}\n")
        log_file.write(f"Scheduler: ReduceLROnPlateau\n")
        log_file.write(f"Model: ModifiedResnet50_Classifier \n")
        log_file.write(f"classifiers: 512, 256, 4 : 2 fully connected layers\n")
        log_file.write(f"########################################################\n\n")
        log_file.write(f"Balanced Accuracy2: {balanced_accuracy2 * 100:.2f}%\n")
        log_file.write(f"Test Accuracy: {accuracy * 100:.2f}%\n")
        log_file.write(f"\nEvaluation Results:\n")
        log_file.write(f"Precision: {precision * 100:.2f}%\n")
        log_file.write(f"Recall: {recall * 100:.2f}%\n")
        log_file.write(f"F1-score: {f1_score * 100:.2f}%\n")
        log_file.write(f"Kappa: {cohens_kappa * 100:.2f}%\n")
        log_file.write(f"########################################################\n\n")
        log_file.write(f"Confusion Matrix:\n")
        log_file.write(f"{conf_mat}\n")
        log_file.write(f"########################################################\n\n")
        log_file.write(f"Classifcation Report: {classification_report(all_true_labels, all_predicted_labels, target_names=true_labels_dic.values(),digits=4)}\n")

    print(classification_report(all_true_labels, all_predicted_labels, target_names=true_labels_dic.values(),digits=4))

    #save this plot as a png file
    plot = plot_confusion_matrix(conf_mat, true_labels_dic.values())
    plot.savefig(f'{dir_path}/confusion_matrix_{measure_checkpoint}.png')


