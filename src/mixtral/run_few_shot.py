import argparse
import replicate
import os
import sys
import json
from tqdm import tqdm
sys.path.append('/home/qnnguyen/stance-detection/code')

from src.mixtral.process_few_shot import process_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prompts and generate predictions.")
    parser.add_argument("--mode", type=str, choices=['emfd', 'frameaxis', 'original'], required=True, help="Processing mode")
    parser.add_argument("--api_token", type=str, required=True, help="API token for Mixtral")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--datasets", type=str, default="semeval", nargs='+', help="List of datasets to process (e.g., semeval pstance cb-dataset)")
    parser.add_argument("--num_shots", type=str, default=1, help="Number of shots")

    args = parser.parse_args()
    
    base_dir = '/home/qnnguyen/stance-detection/code/data'
    
    # Default dataset_targets, can be adjusted based on the datasets specified in the command line
    default_dataset_targets = {
        # 'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        # 'pstance': ['trump', 'bernie', 'biden'],
        # 'cb-dataset': ['trump', 'mask', 'racial']
        'cb-dataset': ['mask']
    }
    
    # Filter dataset_targets based on command line arguments, if any are specified
    if args.datasets:
        dataset_targets = {dataset: targets for dataset, targets in default_dataset_targets.items() if dataset in args.datasets}
    else:
        dataset_targets = default_dataset_targets

    prompt_type = f"{args.mode}/few-shot-test"
    prediction_type = f"{args.mode}/prediction/mixtral-prediction/few-shot/{args.num_shots}-shot"

    process_prompts(args.api_token, base_dir, prompt_type, prediction_type, args.num_experiments, dataset_targets, args.num_shots)

# eMFD: r8_BedSe2G2oxw6Zb922whdhsqNwogxHXv2Npgxq
# frameAxis: r8_Sb8yoCOGu87TUK24UApRkDv0dkV2U7h46BmHK
# Original: r8_BYO1UXyjhjFqBdKk6PGFI85dlE1t9gf25g6UE