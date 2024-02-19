import openai
import os
import json
import sys
import argparse
from tqdm import tqdm
sys.path.append('/home/qnnguyen/stance-detection/code')

from src.gpt.process import process_prompts

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process prompts and generate predictions.")
    parser.add_argument("--mode", type=str, choices=['emfd', 'frameaxis', 'original'], required=True, help="Processing mode")
    parser.add_argument("--api_token", type=str, required=True, help="API token for OpenAI")
    parser.add_argument("--num_experiments", type=int, default=1, help="Number of experiments to run")
    parser.add_argument("--datasets", type=str, default="semeval", nargs='+', help="List of datasets to process (e.g., semeval pstance cb-dataset)")
    parser.add_argument("--num_shots", type=str, default=1, help="Number of shots")
    args = parser.parse_args()
    
    base_dir = '/home/qnnguyen/stance-detection/code/data'
    
    # Default dataset_targets, can be adjusted based on the datasets specified in the command line
    default_dataset_targets = {
        'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        'pstance': ['trump', 'bernie', 'biden'],
        'cb-dataset': ['trump', 'mask', 'racial']
    }
    
    # Filter dataset_targets based on command line arguments, if any are specified
    if args.datasets:
        dataset_targets = {dataset: targets for dataset, targets in default_dataset_targets.items() if dataset in args.datasets}
    else:
        dataset_targets = default_dataset_targets

    # prompt_type = f"{args.mode}/prompt"
    # prediction_type = f"{args.mode}/prediction/gpt-prediction"

    prompt_type = f"{args.mode}/few-shot-test"
    prediction_type = f"{args.mode}/prediction/gpt-prediction/few-shot/{args.num_shots}-shot"

    process_prompts(args.api_token, base_dir, prompt_type, prediction_type, args.num_experiments, dataset_targets)


# original: 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
# frameAxis: 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
# emfd: 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