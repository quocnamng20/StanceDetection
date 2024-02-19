import openai
from openai import OpenAI

import os
import json
import argparse
from tqdm import tqdm

def run_prediction(client, item):
    output = client.completions.create(
        model="gpt-3.5-turbo-instruct", 
        prompt = item['prompt'],
        temperature=0, max_tokens=5, top_p=0.9,
        frequency_penalty=0.0, presence_penalty=0.0
    )
    # print(1)
    # print(output.choices[0].text)
    answer = output.choices[0].text.replace('\n', '').replace('-', '').replace('.', '').replace(' ', '').upper()
    # answer = output['choices'][0]['text'].replace('\n', '').replace('-', '').replace('.', '').replace(' ', '').upper()
    return answer

def process_prompts(api_token, base_dir, prompt_type, prediction_type, num_experiments, dataset_targets):
    client = OpenAI(
        api_key=api_token
    )
    
    for experiment in range(1, num_experiments + 1):
        for dataset, targets in dataset_targets.items():
            for target in targets:
                file_name = f'{target}.json'
                path_prompt = os.path.join(base_dir, dataset, prompt_type, file_name)
                path_prediction = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{file_name}')
                labels = ['SUPPORT', 'AGAINST'] + (['NONE'] if dataset == 'semeval' else [])
                try:
                    with open(path_prompt, 'r', encoding='utf-8') as file:
                        prompt = json.load(file)
                    
                    for i, item in tqdm(enumerate(prompt), desc=f'Experiment {experiment} - {dataset} - {target}'):
                        # output = client.completions.create(
                        #     model="gpt-3.5-turbo-instruct", 
                        #     prompt = item['prompt'],
                        #     temperature=0, max_tokens=5, top_p=0.9,
                        #     frequency_penalty=0.0, presence_penalty=0.0
                        # )
                        # # print(1)
                        # # print(output.choices[0].text)
                        # answer = output.choices[0].text.replace("'", '').upper()
                        answer = run_prediction(client, item)
                        if answer not in labels:
                            answer = run_prediction(client, item)
                        item['response'] = answer
                    
                    with open(path_prediction, 'w', encoding='utf-8') as file:
                        json.dump(prompt, file, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f"Error processing {dataset} - {target}: {e}")