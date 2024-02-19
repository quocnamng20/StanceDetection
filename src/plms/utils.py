# utils.py
import pandas as pd
from sklearn.preprocessing import StandardScaler
from transformers import AutoTokenizer
from torch.utils.data import Dataset
import torch

def map_label(label, dataset_name):
    if dataset_name == 'semeval':
        return {'FAVOR': 1, 'AGAINST': 0, 'NONE': 2}.get(label, label)
    elif dataset_name == 'cb-dataset':
        return {'SUPPORT': 1, 'AGAINST': 0}.get(label, label)
    else:
        return {'FAVOR':1, 'AGAINST':0}.get(label, label)

class CustomDataset(Dataset):
    def __init__(self, encodings, labels, numerical_features=None):
        self.encodings = encodings
        self.labels = labels
        self.numerical_features = numerical_features

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.numerical_features is not None:
            item['numerical_features'] = torch.tensor(self.numerical_features[idx], dtype=torch.float32)
        return item

    def __len__(self):
        return len(self.labels)

def prepare_data(df, tokenizer, dataset_name, feature_type=None):
    """
    Prepares data for model training or evaluation.

    Args:
    - df (pandas.DataFrame): The DataFrame containing the data.
    - tokenizer (transformers.AutoTokenizer): Tokenizer for text data.
    - dataset_name (str): Name of the dataset to determine label mapping.
    - feature_type (str, optional): Type of numerical features to include ('emfd' or 'frameaxis').

    Returns:
    - CustomDataset: A dataset ready for training or evaluation.
    """
    # Map labels based on dataset_name
    # print(f"DataFrame type: {type(df)}")
    # sample_label = 'FAVOR'
    # print(map_label(sample_label, 'semeval'))  # Expected output: 1

    # sample_label = 'SUPPORT'
    # print(map_label(sample_label, 'cb-dataset'))

    # print(type(df['Stance']))
    # print(df.head())
    df['label'] = df['Stance'].apply(lambda x: map_label(x, dataset_name))

    # Tokenize texts
    texts = df['Tweet'].tolist()
    labels = df['label'].tolist()
    encodings = tokenizer(texts, truncation=True, padding=True, max_length=512)

    numerical_features = None
    if feature_type and feature_type in ['emfd', 'frameaxis']:
        feature_columns = {
            'emfd': [col for col in df.columns if 'eMFD' in col],
            'frameaxis': [col for col in df.columns if 'FrameAxis' in col],
        }.get(feature_type, [])
        if feature_columns:  # Proceed if the feature columns list is not empty
            scaler = StandardScaler()
            numerical_features = scaler.fit_transform(df[feature_columns].values)
    else:
        print(f"No numerical features of type '{feature_type}' found.")

    return CustomDataset(encodings, labels, numerical_features)