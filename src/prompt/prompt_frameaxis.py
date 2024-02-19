import replicate

import os
import json
import nltk
import sys
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
sys.path.append('/home/qnnguyen/stance-detection/code')

from src.utils.frameaxis import create_humanized_prompt_with_explanations
from src.utils.kmeans_improve import compute_improved_kmeans_thresholds

tqdm.pandas(desc="Processing rows")
base_dir = '/home/qnnguyen/stance-detection/code/data'
mode = 'train'
prompt_path = f'frameaxis/few-shot-{mode}'
prompt_type = 'few-shot'
# Mapping datasets to their respective targets
dataset_targets = {
    'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
    'pstance': ['trump', 'bernie', 'biden'],
    'cb-dataset': ['trump', 'mask', 'racial']
}

# Update FrameAxis columns
bias_columns = ['bias_loyalty', 'bias_care', 'bias_fairness', 'bias_authority', 'bias_sanctity']
intensity_columns = ['intensity_loyalty', 'intensity_care', 'intensity_fairness', 'intensity_authority', 'intensity_sanctity']

# Iterate over datasets and their targets
for dataset, targets in dataset_targets.items():
  for target in targets:
    # Construct the file path dynamically
    file_name = f'{mode}_{target}_frameaxis.csv' if dataset in ['semeval', 'pstance'] else f'{target}_{mode}.csv'
    path = os.path.join(base_dir, dataset, file_name)
    
    # Read the dataset
    df = pd.read_csv(path)
    
    # Calculate thresholds
    bias_thresholds = compute_improved_kmeans_thresholds(df, bias_columns)
    intensity_thresholds = compute_improved_kmeans_thresholds(df, intensity_columns)
    
    # Apply the function to create humanized prompts
    df['prompt'] = df.progress_apply(lambda row: create_humanized_prompt_with_explanations(row, bias_thresholds, intensity_thresholds, dataset, prompt_type), axis=1)
    
    if mode == 'train':
      for index, row in df.iterrows():
          row['prompt'] = row['prompt'] +  f"""\nStance: {row['Stance']}"""
    
    # Define the output JSON file path
    json_file_path = os.path.join(base_dir, dataset, prompt_path, f'{target}.json')
    
    # Save the DataFrame as a JSON file
    df.to_json(json_file_path, orient='records')
