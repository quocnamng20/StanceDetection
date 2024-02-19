import os
import json
import nltk
import sys
import random
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm
import argparse

# tqdm.pandas(desc="Processing rows") # This line can cause issue in script, please uncomment if needed

def process_data(dataset_names, num_few_shot, mode):
    base_dir = '/home/qnnguyen/stance-detection/code/data' # Hardcoded base_dir
    test_mode = 'test' # Hardcoded test_mode
    train_mode = 'train' # Hardcoded train_mode
    
    dataset_targets = {
        'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        'pstance': ['trump', 'bernie', 'biden'],
        'cb-dataset': ['trump', 'mask', 'racial']
    }
    
    for dataset_name in dataset_names:
        if dataset_name not in dataset_targets:
            print(f"Dataset '{dataset_name}' not recognized.")
            continue
        
        targets = dataset_targets[dataset_name]
        
        for target in targets:
            # Construct the file path dynamically
            test_path = os.path.join(base_dir, dataset_name, f'{mode}/few-shot-{test_mode}', f'{target}.json')
            train_path = os.path.join(base_dir, dataset_name, f'{mode}/few-shot-{train_mode}', f'{target}.json')

            with open(test_path, 'r') as file:
                test_data = json.load(file)

            with open(train_path, 'r') as file:
                train_data = json.load(file)

            if dataset_name == 'semeval':
                beginning = f"""The following tweets are social media tweets about {target}. The user's stance towards the tweet can support, against, or none. The stances can be described as:
                - support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
                - against: The tweet opposes or criticizes the target, either explicitly or implicitly.
                - none: The tweet is neutral, doesn’t have a stance towards the target or unrelated to the target.\n"""

                ending = f"""\nNow, classify user's stance as to whether 'SUPPORT', 'AGAINST', or 'NONE' towards the following tweet. Only return the classification for the user's stance, and no other text.\n"""
            else:
                beginning = f"""The following tweets are social media tweets about {target}. The user's stance towards the tweet can support, against, or none. The stances can be described as:
                - support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
                - against: The tweet opposes or criticizes the target, either explicitly or implicitly.\n"""

                ending = f"""\nNow, classify user's stance as to whether 'SUPPORT' or 'AGAINST' towards the following tweet. Only return the classification for the user's stance, and no other text.\n"""

            for i in range(len(test_data)): # Fix range issue

                random_ids = random.sample(range(1, len(train_data)), num_few_shot)

                few_shot = ''
                for id in random_ids:
                    few_shot += train_data[id]['prompt'] + '\n' # Fix this line

                test_shot = test_data[i]['prompt']
                question = '\nStance:'
                new_prompt = beginning + '\n' + few_shot + ending + test_shot + question 

                test_data[i]['prompt'] = new_prompt

            test_path_shot = os.path.join(base_dir, dataset_name, f'{mode}/few-shot-{test_mode}', f'{target}_{num_few_shot}_shot.json')
            with open(test_path_shot, 'w') as file:
                json.dump(test_data, file)

def main(args):
    dataset_names = args.dataset_names
    num_few_shot = args.num_few_shot
    mode = args.mode

    process_data(dataset_names, num_few_shot, mode)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process data for few-shot learning")
    parser.add_argument("--dataset_names", nargs='+', type=str, help="Names of the datasets")
    parser.add_argument("--num_few_shot", type=int, help="Number of few shot")
    parser.add_argument("--mode", type=str, help="Mode of interpretation")
    args = parser.parse_args()

    main(args)
