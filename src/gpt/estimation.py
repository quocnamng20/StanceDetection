import openai
import pandas as pd
import replicate

import os
import json
import sys
import nltk
import tiktoken
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
sys.path.append('/home/qnnguyen/stance-detection/code')

api_original = 'r8_T7SfjUY7ZSDIkLU7JzcDyzMFAOeP8nW2r0xL3'
replicate = replicate.client.Client(api_token=api_original)

# Iterate over datasets and their targets

num_experiments = 1

dataset_targets = {
    # 'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
    # 'pstance': ['trump', 'bernie', 'biden'],
    'cb-dataset': ['trump', 'mask', 'racial']
}

modes = ['original', 'emfd', 'frameaxis']

def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

# print(num_tokens_from_string("Hello world, let's test tiktoken.", "cl100k_base"))
for mode in modes:
    base_dir = '/home/qnnguyen/stance-detection/code/data'
    prompt_type = f'{mode}/few-shot-test'
    prediction_type = f'{mode}/prediction'
    total_tokens_emfd = 0
    for experiment in range(1, num_experiments + 1):
        total_tokens = 0
        tokens_ = []
        for dataset, targets in dataset_targets.items():
            for target in targets:
                file_name = f'{target}.json'
                path_prompt = os.path.join(base_dir, dataset, prompt_type, file_name)
                path_prediction = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{file_name}')
                
                # Load the prompt data
                with open(path_prompt, 'r', encoding='utf-8') as file:
                    prompt = json.load(file)
                count = 0
                # # Process each item in the prompt
                for i, item in tqdm(enumerate(prompt)):
                    tokens = num_tokens_from_string(item['prompt'], 'cl100k_base')
                    tokens_.append(tokens)
                    if tokens >= 4096:
                        count += 1
                print(f'{dataset}_{target}_{mode}: {count}')
                plt.figure(figsize=(10, 6))
                plt.hist(tokens_, bins=30, color='skyblue', edgecolor='black')
                plt.title('Distribution of Token Counts')
                plt.xlabel('Token Count')
                plt.ylabel('Frequency')
                plt.grid(axis='y', alpha=0.75)

                # plt.savefig(f'/home/qnnguyen/stance-detection/code/data/plot/{dataset}_{target}_{mode}_5_shot_distribution_of_token_counts.png')

                    # total_tokens += tokens
                # print(f'{dataset} - {target}-{total_tokens}')
            # if dataset == 'cb-dataset':
                # total_tokens /= 5
            # total_tokens_emfd += total_tokens
    # print(f'{total_tokens_emfd} - {mode}')

