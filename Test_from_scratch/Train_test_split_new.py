import pandas as pd
from sklearn.model_selection import train_test_split


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

