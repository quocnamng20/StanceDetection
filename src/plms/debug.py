import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback, LongformerTokenizer

# Ensure CUDA is available
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# Append the necessary path for custom modules
sys.path.append('/home/qnnguyen/stance-detection/code/')
from src.plms.utils import prepare_data

def filter_long_sequences(dataset_path, max_length=512):
    """
    Filters out samples from a dataset where the tokenized text length exceeds a specified maximum.

    Args:
    - dataset_path (str): Path to the dataset CSV file.
    - max_length (int): Maximum allowed tokenized text length.

    Returns:
    - filtered_df (pandas.DataFrame): The filtered DataFrame.
    """
    # Load the dataset
    df = pd.read_csv(dataset_path)
    
    # Define a function to tokenize text and check its length
    def is_within_limit(text):
        # Tokenize the text and check if its length is within the specified limit
        encoded_text = tokenizer.encode(text, add_special_tokens=True)
        return len(encoded_text) <= max_length
    
    # Apply the function to filter the dataset
    filtered_df = df[df['Tweet'].apply(is_within_limit)]
    
    # Save the filtered dataset to a new file
    # filtered_dataset_path = dataset_path.replace('.csv', f'_filtered.csv')
    # filtered_df.to_csv(filtered_dataset_path, index=False)
    
    # print(f"Filtered dataset saved to {filtered_dataset_path}")
    return filtered_df

targets = ['racial', 'mask', 'trump']
tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")

for target in targets:
    max_length = 0
    count = 0
    df_train = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/data/{target}_train.csv'
    df_test = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/data/{target}_test.csv'
    # Iterate over the dataset and tokenize each text entry
    # tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    
    # for text in df_train['Tweet']:  # Assume your text column is named 'Tweet'
    #     encoded_text = tokenizer.encode(text, add_special_tokens=True)  # Encode the text
    #     length = len(encoded_text)  # Get the length of the encoded text
    #     if length > max_length:  # Update max_length if the current length is greater
    #         max_length = length
    #     if length > 4096:
    #         count +=1 

    # print(f"The maximum sequence length in the dataset is: {max_length}")
    # print(f'Number of samples: {count}')

    filtered_df_train = filter_long_sequences(df_train, 512)
    filtered_df_test = filter_long_sequences(df_test, 512)
    print(f"{target} dataset filtered. Remaining samples: {len(filtered_df_train)}")
    print(f"{target} dataset filtered. Remaining samples: {len(filtered_df_test)}")

