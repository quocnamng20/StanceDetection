import replicate

import pandas as pd
import numpy as np
import argparse
import os
import sys
import nltk
import json
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
nltk.download('punkt')

def standardize_responses(base_dir, prediction_type, num_experiments, dataset_targets):
    """
    Standardizes the stance terminology in JSON prediction files by replacing "SUPPORT" with "FAVOR".
    """
    for experiment in range(1, num_experiments + 1):
        for dataset, targets in dataset_targets.items():
            for target in targets:
                file_path = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{target}.json')
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as file:
                        predictions = json.load(file)
                except FileNotFoundError:
                    print(f"File not found: {file_path}")
                    continue
                
                if dataset != 'cb-dataset':
                    for item in tqdm(predictions, desc=f'Experiment {experiment} - {dataset} - {target}'):
                        if item['response'] == 'SUPPORT' or item['response'] == 'SUP': 
                            item['response'] = 'FAVOR'
                        elif item['response'] == 'AGA':
                            item['response'] = 'AGAINST'
                else:
                    for item in tqdm(predictions, desc=f'Experiment {experiment} - {dataset} - {target}'):
                        if item['response'] == 'SUP': 
                            item['response'] = 'SUPPORT'
                        elif item['response'] == 'AGA':
                            item['response'] = 'AGAINST'
                modified_file_path = file_path.replace('.json', '_modified.json')
                with open(modified_file_path, 'w', encoding='utf-8') as file:
                    json.dump(predictions, file, indent=4)

def calculate_f1_scores(base_dir, prediction_type, num_experiments, dataset_targets, mode):
    """
    Calculates and prints the mean F1 macro score for each target within each dataset, 
    and also calculates and prints the mean F1 macro score for each dataset as a whole, 
    across multiple experiments.
    """
    overall_scores = {}  # Structure to hold scores for aggregation

    for experiment in range(1, num_experiments + 1):
        for dataset, targets in dataset_targets.items():
            if dataset not in overall_scores:
                overall_scores[dataset] = {target: [] for target in targets}
                overall_scores[dataset]['_dataset_mean'] = []  # Special key for dataset-wide mean scores
            
            for target in targets:
                file_path = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{target}_modified.json')
                
                try:
                    df = pd.read_json(file_path)
                except Exception as e:  # Broad exception handling for any file read or JSON parsing issue
                    print(f"Error processing file {file_path}: {e}")
                    continue
                
                if 'Stance' in df.columns and 'response' in df.columns:
                    score = f1_score(df['Stance'], df['response'], average='macro')
                    overall_scores[dataset][target].append(score)
                    overall_scores[dataset]['_dataset_mean'].append(score)  # Add score to dataset-wide mean
                else:
                    print(f"Missing required columns in {file_path}")
    
    # After collecting scores, calculate and print mean F1 score for each target and dataset
    for dataset, targets_scores in overall_scores.items():
        for target, scores in targets_scores.items():
            if target == '_dataset_mean':
                continue  # Skip the dataset-wide mean here, handle it separately
            
            if scores:  # Ensure there are scores to calculate mean
                mean_score = np.mean(scores)
                # print(f'{dataset} - {target} - {mode}: Mean F1 Score across {num_experiments} experiments: {scores}')
                print(f'{dataset} - {target} - {mode}: Mean F1 Score across {num_experiments} experiments: {mean_score}')
        
        # Now handle the dataset-wide mean F1 score
        dataset_mean_scores = targets_scores['_dataset_mean']
        if dataset_mean_scores:
            mean_dataset_score = np.mean(dataset_mean_scores)
            print(f'{dataset} - {mode}: Overall Mean F1 Score across {num_experiments} experiments: {mean_dataset_score}')
        else:
            print(f'{dataset} - {mode}: No F1 Scores available for overall dataset')

    # f1 = []
    # for target in targets:
    # file_path = f'/content/cb-dataset/emfd/prediction/ver1/{target}.json'
    # df = pd.read_json(file_path)

    # if 'Stance' in df.columns and 'response' in df.columns:
    #     # Calculate F1-macro score

    #     f1_macro = f1_score(df['Stance'], df['response'], average='macro')
    #     f1_macro
    # else:
    #     f1_macro = None
    # f1.append(f1_macro)
    # print(np.mean(f1))
    # fig = plt.figure(figsize = (10, 5))

    # # creating the bar plot
    # plt.bar(targets, f1, color ='green',
    #         width = 0.4)

    # plt.xlabel(f"CB-eMFD")
    # plt.ylabel("F1 score")

    # plt.show()
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze stance detection predictions.")
    parser.add_argument("--mode", type=str, required=True, choices=['emfd', 'frameaxis', 'original'], help="Processing mode.")
    parser.add_argument("--model", type=str, help="Replicate model for prompting", required=True, choices=['llama', 'mixtral', 'gpt'])
    args = parser.parse_args()
    
    base_dir = '/home/qnnguyen/stance-detection/code/data'
    num_experiments = 5
    dataset_targets = {
        'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        'pstance': ['trump', 'bernie', 'biden'],
        'cb-dataset': ['trump', 'mask', 'racial']
    }

    # prompt_type = f"{args.mode}/prompt"
    prediction_type = f"{args.mode}/prediction/{args.model}-prediction"

    standardize_responses(base_dir, prediction_type, num_experiments, dataset_targets)
    calculate_f1_scores(base_dir, prediction_type, num_experiments, dataset_targets, args.mode)