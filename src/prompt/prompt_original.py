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

from src.utils.original import create_humanized_prompt_with_explanations

tqdm.pandas(desc="Processing rows")
mode = 'test'
base_dir = '/home/qnnguyen/stance-detection/code/data'
prompt_path = f'original/few-shot-{mode}'

# Mapping datasets to their respective targets
dataset_targets = {
    'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
    'pstance': ['trump', 'bernie', 'biden'],
    'cb-dataset': ['trump', 'mask', 'racial']
}
prompt_type = 'few-shot'
# Iterate over datasets and their targets
for dataset, targets in dataset_targets.items():
    for target in targets:
        # Construct the file path dynamically
        file_name = f'{mode}_{target}_emfd.csv' if dataset in ['semeval', 'pstance'] else f'{target}_{mode}.csv'
        path = os.path.join(base_dir, dataset, file_name)
        
        # Read the dataset
        df = pd.read_csv(path)
        
        # Apply the function to create humanized prompts
        df['prompt'] = df.progress_apply(lambda row: create_humanized_prompt_with_explanations(row, dataset, prompt_type), axis=1)
        
        if mode == 'train':
            for index, row in df.iterrows():
                update_prompt = row['prompt'] +  f"""\nStance: {row['Stance']}"""
                df.loc[index, 'prompt'] = update_prompt
        
        json_file_path = os.path.join(base_dir, dataset, prompt_path, f'{target}.json')
        
        # Save the DataFrame as a JSON file
        df.to_json(json_file_path, orient='records')
