import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import load_metric
from transformers import AutoModelForSequenceClassification, TrainingArguments, Trainer, AutoTokenizer, EarlyStoppingCallback

# Ensure CUDA is available
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 

# Append the necessary path for custom modules
sys.path.append('/home/qnnguyen/stance-detection/code/')
from src.plms.utils import prepare_data

# Load the F1 metric
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    f1_result = f1_metric.compute(predictions=predictions, references=labels, average="macro")
    f1_score = f1_result['f1']
    return {"f1": f1_score}

def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for stance detection")
    parser.add_argument("--dataset_name", type=str, nargs='+', required=True, help="Name of the dataset to use")
    parser.add_argument("--feature_type", type=str, default=None, help="Type of features to use")
    parser.add_argument("--model_path", type=str, default="vinai/bertweet-base", help="Model path for tokenizer and model")
    parser.add_argument("--num_epochs", type=int, default=30, help="Number of epochs for training")
    parser.add_argument("--batch_size", type=int, default=16, help="Training batch size per device")
    parser.add_argument("--val_split_percentage", type=float, default=0.1, help="Validation split percentage")
    return parser.parse_args()

def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    default_dataset_targets = {
        'semeval': ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion'],
        'pstance': ['trump', 'bernie', 'biden'],
        'cb-dataset': ['trump', 'mask', 'racial']
    }

    for dataset_name in args.dataset_name:
        for target in default_dataset_targets.get(dataset_name, []):
            print(f"Processing target: {target} for dataset: {dataset_name}")
            
            # Adjust dataset loading based on feature_type
            csv_suffix = f"_{args.feature_type}" if args.feature_type else ""
            target_df = pd.read_csv(f'/home/qnnguyen/stance-detection/code/data/{dataset_name}/plms/data/{target}_train{csv_suffix}.csv')
        
            # Split the dataset
            train_df, val_df = train_test_split(target_df, test_size=args.val_split_percentage)
            train_dataset = prepare_data(train_df, tokenizer, dataset_name, args.feature_type)
            val_dataset = prepare_data(val_df, tokenizer, dataset_name, args.feature_type)

            # Initialize model
            model = AutoModelForSequenceClassification.from_pretrained(args.model_path, num_labels=2)  # Assuming 3 classes for simplification
            
            # Define training arguments with corrected output_dir path
            output_dir_path = f"./results/{dataset_name}/{args.feature_type}/{target}"
            training_args = TrainingArguments(
                output_dir=output_dir_path,
                num_train_epochs=args.num_epochs,
                per_device_train_batch_size=args.batch_size,
                per_device_eval_batch_size=args.batch_size,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                weight_decay=0.01,
                load_best_model_at_end=True,
                metric_for_best_model='f1',
                save_total_limit=5,
            )

            # Instantiate Trainer
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=val_dataset,
                compute_metrics=compute_metrics,
                callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
            )

            # Start training
            trainer.train()

if __name__ == "__main__":
    main()
