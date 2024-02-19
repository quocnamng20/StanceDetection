import replicate

import os
import json
import sys
import nltk
import pandas as pd
import numpy as np
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
base_dir = '/home/qnnguyen/stance-detection/code/data'
prompt_type = 'emfd/prompt'
prediction_type = 'emfd/prediction'
num_experiments = 5

dataset_targets = {
    'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
    'pstance': ['trump', 'bernie', 'biden'],
    'cb-dataset': ['trump', 'mask', 'racial']
}

for experiment in range(1, num_experiments + 1):
    for dataset, targets in dataset_targets.items():
        for target in targets:
            file_name = f'{target}.json'
            path_prompt = os.path.join(base_dir, dataset, prompt_type, file_name)
            path_prediction = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{file_name}')
            
            # Load the prompt data
            with open(path_prompt, 'r', encoding='utf-8') as file:
                prompt = json.load(file)
            
            # Process each item in the prompt
            for i, item in tqdm(enumerate(prompt), desc=f'Experiment {experiment} - {dataset} - {target}'):
                output = replicate.run(
                    "meta/llama-2-70b-chat",
                    input={"prompt": item['prompt'],
                           "temperature": 0.75, "top_p": 0.9, "top_k": 10,
                           "max_new_tokens": 3, "min_new_tokens": -1,
                           "debug": True,
                           }
                )
                out = list(output)
                answer = ''.join(out).replace('\n', '').replace('- ', '').replace('.', '').upper()

                item['response'] = answer

            # Save the updated prompt with responses
            with open(path_prediction, 'w', encoding='utf-8') as file:
                json.dump(prompt, file, ensure_ascii=False, indent=4)