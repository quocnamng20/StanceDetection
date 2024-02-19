import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.preprocessing import StandardScaler

# A function to map labels to integers, if necessary
def map_label(label, dataset_name):
    if dataset_name == 'semeval':
        return {'FAVOR': 1, 'AGAINST': 0, 'NONE': 2}.get(label, label)
    elif dataset_name == 'cb-dataset':
        return {'SUPPORT': 1, 'AGAINST': 0}.get(label, label)
    else:
        return {'FAVOR':1, 'AGAINST':0}.get(label, label)

# Custom Dataset for loading user-level embeddings
class UserDataset(Dataset):
    def __init__(self, embeddings, labels):
        self.embeddings = embeddings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "embedding": torch.tensor(self.embeddings[idx], dtype=torch.float),
            "label": torch.tensor(self.labels[idx], dtype=torch.long)
        }

def prepare_data(file_path, tokenizer, model, dataset_name, label_column='Stance', text_column='Tweet'):
    df = pd.read_csv(file_path)
    df['label_mapped'] = df[label_column].apply(lambda x: map_label(x, dataset_name))
    embeddings = []

    # Generate embeddings for each text
    for _, row in tqdm(df.iterrows()):
        inputs = tokenizer(row[text_column], return_tensors="pt", padding=True, truncation=True, max_length=128)
        outputs = model(**inputs)
        # Use the CLS token's embedding
        embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().detach().numpy())
    print(3)
    # Standardize embeddings
    scaler = StandardScaler()
    standardized_embeddings = scaler.fit_transform(np.vstack(embeddings))

    labels = df['label_mapped'].values
    return standardized_embeddings, labels
