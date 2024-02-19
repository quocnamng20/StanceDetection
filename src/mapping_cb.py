import pandas as pd

targets = ['trump', 'mask', 'racial']

for target in targets:
    df = pd.read_csv(f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/tweet_level/{target}_test.csv')
    # stance_mapping = {
    #     0: 'SUPPORT',
    #     1: 'AGAINST'
    # }
    # df = df.rename(columns={'tweet': 'Tweet', 'stance': 'Stance'})
    # for stance_key, stance_value in stance_mapping.items():
    #     df['Stance'] = df['Stance'].replace(stance_key, stance_value)

    stance_mapping = {
        'mask': 'Wearing Mask',
        'trump': 'Donald Trump',
        'racial': 'Racial Equality'
    }
    if target == 'mask':
        target_ = 'Wearing Mask'
    elif target == 'trump':
        target_ = 'Donald Trump'
    else:
        target_ = 'Racial Equality'
    df['Target'] = target_
    df.to_csv(f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/data/{target}_test.csv')