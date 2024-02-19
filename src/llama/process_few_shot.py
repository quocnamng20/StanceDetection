import replicate
import os
import json
from tqdm import tqdm

# Function to process prompts
def process_prompts(api_token, base_dir, prompt_type, prediction_type, num_experiments, dataset_targets, num_shots):
    replicate_client = replicate.client.Client(api_token=api_token)
    
    for experiment in range(1, num_experiments + 1):
        for dataset, targets in dataset_targets.items():
            labels = ['SUPPORT', 'AGAINST', 'SUP', 'AGA'] + (['NONE'] if dataset == 'semeval' else [])
            for target in targets:
                file_name = f'{target}_{num_shots}_shot.json'
                path_prompt = os.path.join(base_dir, dataset, prompt_type, file_name)
                path_prediction = os.path.join(base_dir, dataset, prediction_type, f'experiment_{experiment}_{file_name}')
                
                try:
                    with open(path_prompt, 'r', encoding='utf-8') as file:
                        prompt = json.load(file)
                    
                    for i, item in tqdm(enumerate(prompt), desc=f'Experiment {experiment} - {dataset} - {target}'):
                        output = replicate_client.run(
                            "meta/llama-2-70b-chat",
                            input={"prompt": item['prompt'],
                                "temperature": 0.01, "top_p": 0.9, "top_k": 10,
                                "max_new_tokens": 3, "min_new_tokens": -1,
                                "debug": True,
                                }
                        )
                        out = list(output)
                        answer = ''.join(out).replace('\n', '').replace('- ', '').replace('.', '').upper()
                        item['response'] = answer
                    
                    with open(path_prediction, 'w', encoding='utf-8') as file:
                        json.dump(prompt, file, ensure_ascii=False, indent=4)
                except Exception as e:
                    print(f'Error processing {dataset} - {target}: {e}')

# def f1-score(targets):
#     f1 = []
#     for target in targets:
#     file_path = f'/content/cb-dataset/emfd/prediction/ver1/{target}.json'
#     df = pd.read_json(file_path)

#     if 'Stance' in df.columns and 'response' in df.columns:
#         # Calculate F1-macro score

#         f1_macro = f1_score(df['Stance'], df['response'], average='macro')
#         f1_macro
#     else:
#         f1_macro = None
#     f1.append(f1_macro)
#     print(np.mean(f1))
#     fig = plt.figure(figsize = (10, 5))

#     # creating the bar plot
#     plt.bar(targets, f1, color ='green',
#             width = 0.4)

#     plt.xlabel(f"CB-eMFD")
#     plt.ylabel("F1 score")

#     plt.show()`