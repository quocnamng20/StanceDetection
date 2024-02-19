import replicate
import os
import json
from tqdm import tqdm

def run_prediction(client, item, experiment_params):
    output = client.run(
        "mistralai/mixtral-8x7b-instruct-v0.1",
        input={
            "prompt": item['prompt'],
            **experiment_params
        }
    )
    return ''.join(list(output)).translate(str.maketrans('', '', '\n- .\xa0\t🟢✅ **✔️')).upper()

def process_prompts(api_token, base_dir, prompt_type, prediction_type, num_experiments, dataset_targets):
    replicate_client = replicate.Client(api_token=api_token)
    base_params = {
        "temperature": 0, "top_p": 0.9, "top_k": 10,
        "max_new_tokens": 3,
        "prompt_template": "<s>[INST] {prompt}. Please do not response any additional reasoning information and any strange character. [/INST]",
        "presence_penalty": 0, "frequency_penalty": 0, "debug": True,
    }

    for experiment in range(1, num_experiments + 1):
        for dataset, targets in dataset_targets.items():
            labels = ['SUPPORT', 'AGAINST'] + (['NONE'] if dataset == 'semeval' else [])
            for target in targets:
                path_prompt = os.path.join(base_dir, dataset, prompt_type, f'{target}.json')
                path_prediction = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{target}.json')
                
                try:
                    with open(path_prompt, 'r', encoding='utf-8') as file:
                        prompt = json.load(file)

                    for i, item in tqdm(enumerate(prompt), desc=f'Experiment {experiment} - {dataset} - {target}'):
                        answer = run_prediction(replicate_client, item, base_params)
                        if answer not in labels:
                            # Adjust parameters for a second attempt if necessary
                            answer = run_prediction(replicate_client, item, {**base_params, "top_k": 50, "max_new_tokens": 2})

                        item['response'] = answer

                    with open(path_prediction, 'w', encoding='utf-8') as file:
                        json.dump(prompt, file, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error processing {dataset} - {target}: {e}")
