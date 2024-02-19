# evaluate.py
from transformers import AutoModelForSequenceClassification, Trainer
from utils import prepare_data, CustomDataset
from transformers import AutoTokenizer
from datasets import load_metric
import pandas as pd
import numpy as np
f1_metric = load_metric("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    return {"f1": f1_metric.compute(predictions=predictions, references=labels, average="macro")}

def main():
    target = 'bernie'
    feature_type = 'emfd'  # 'FrameAxis', None
    dataset_name = 'pstance'  # or 'pstance', 'cb-dataset'
    # root = f"./results/pstance/{feature_type}"
    model_path = f"./results/{dataset_name}/{feature_type}/{target}/checkpoint-729"  # Adjust to your checkpoint
    tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base")
    if feature_type != None:
        test_file = pd.read_csv(f'/home/qnnguyen/stance-detection/code/data/{dataset_name}/plms/data/{target}_test_{feature_type}.csv')
    else:
        test_file = pd.read_csv(f'/home/qnnguyen/stance-detection/code/data/{dataset_name}/plms/data/{target}_test.csv')

    test_dataset = prepare_data(test_file, tokenizer, dataset_name, feature_type)

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    trainer = Trainer(
        model=model,
        compute_metrics=compute_metrics,
    )

    results = trainer.evaluate(test_dataset)
    print(target, dataset_name, feature_type)
    print(results)

if __name__ == "__main__":
    main()
