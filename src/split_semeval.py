import pandas as pd

targets = ['atheism', 'climate', 'trump', 'feminist', 'hillary', 'abortion']
semeval_targets = {
    'atheism': 'Atheism',
    'climate': 'Climate Change is a Real Concern',
    'trump': 'Donald Trump',
    'feminist': 'Feminist Movement',
    'hillary': 'Hillary Clinton',
    'abortion': 'Legalization of Abortion'
  }
# df = pd.read_csv('/home/qnnguyen/stance-detection/code/data/semeval/semEval2016_train_w_frameAxis.csv')
# for target_key, target_value in semeval_targets.items():
#   df[df['Target'] == target_value].to_csv(f'/home/qnnguyen/stance-detection/code/data/semeval/train_{target_key}_frameaxis.csv', index=False)

df = pd.read_csv('/home/qnnguyen/stance-detection/code/data/semeval/plms/semEval2016_test_w_frameAxis.csv')
for target_key, target_value in semeval_targets.items():
  df[df['Target'] == target_value].to_csv(f'/home/qnnguyen/stance-detection/code/data/semeval/plms/data/{target_key}_test_frameaxis.csv', index=False)