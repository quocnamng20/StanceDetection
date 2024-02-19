import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import os
import sys
import numpy as np
import pandas as pd
from datasets import load_metric
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback

# Ensure CUDA is available
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# Append the necessary path for custom modules
sys.path.append('/home/qnnguyen/stance-detection/code/')
from src.plms.cb_dataset.utils import UserDataset, prepare_data

class SimpleClassifier(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleClassifier, self).__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.fc(x)

def train(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(train_loader):
        embeddings = batch['embedding'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        outputs = model(embeddings)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(train_loader)

def evaluate(model, val_loader, device):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for batch in tqdm(val_loader):
            embeddings = batch['embedding'].to(device)
            labels = batch['label'].to(device)
            outputs = model(embeddings)
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy = 100 * (np.array(all_predictions) == np.array(all_labels)).mean()
    f1 = f1_score(all_labels, all_predictions, average='macro')
    return accuracy, f1

def parse_args():
    parser = argparse.ArgumentParser(description="Train and evaluate BERTweet-based model")
    parser.add_argument("--dataset_names", nargs="+", required=True,
                        help="List of dataset names to process")
    parser.add_argument("--feature_type", type=str, default=None,
                        help="Type of features to include in the training")
    # Add other command-line arguments as needed
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", normalization=True)
    bertweet_model = AutoModel.from_pretrained("vinai/bertweet-base")

    default_dataset_targets = {
        'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        'pstance': ['trump', 'bernie', 'biden'],
        'cb-dataset': ['trump', 'mask', 'racial']
    }

    for dataset_name in args.dataset_names:
        for target in default_dataset_targets.get(dataset_name, []):
            print(f"Processing target: {target} for dataset: {dataset_name}")
            # csv_suffix = f"_{args.feature_type}" if args.feature_type else ""
            # target_df = pd.read_csv(f'/home/qnnguyen/stance-detection/code/data/{dataset_name}/plms/data/{target}_train{csv_suffix}.csv')
        
            # Adjust dataset loading based on feature_type
            csv_suffix = f"_{args.feature_type}" if args.feature_type else ""
            print(1)
            file_path = f'/home/qnnguyen/stance-detection/code/data/{dataset_name}/plms/data/{target}_train{csv_suffix}_top20000.csv'
            embeddings, labels = prepare_data(file_path, tokenizer, bertweet_model, dataset_name)
            print(2)
            # Split the dataset
            X_train, X_val, y_train, y_val = train_test_split(embeddings, labels, test_size=0.2, random_state=42)

            # Prepare datasets and dataloaders
            train_dataset = UserDataset(X_train, y_train)
            val_dataset = UserDataset(X_val, y_val)
            train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

            model = SimpleClassifier(input_dim=768, num_classes=2).to(device)  # Adjust num_classes as per your dataset
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(model.parameters(), lr=1e-4)

            epochs = 10
            for epoch in range(epochs):
                print(f'Epoch: {epoch}')
                train_loss = train(model, train_loader, criterion, optimizer, device)
                val_accuracy, val_f1 = evaluate(model, val_loader, device)
                print(f'Epoch {epoch+1}, Train Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%, Val F1-Score: {val_f1:.4f}')

if __name__ == "__main__":
    main()
