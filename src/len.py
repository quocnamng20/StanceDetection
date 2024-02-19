import pandas as pd

targets = ['mask', 'racial', 'trump']

modes = ['train', 'test']
for target in targets:
    for mode in modes:
        path = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/data/{target}_{mode}.csv'

        df = pd.read_csv(path)
        # Define the columns for each file based on assumed criteria
        # print(path, df.columns)
        basic_columns = ['user_id', 'tweet', 'stance', 'Target']
        emfd_columns = basic_columns + [
            'care_p', 'fairness_p', 'loyalty_p', 'authority_p', 'sanctity_p',
            'care_sent', 'fairness_sent', 'loyalty_sent', 'authority_sent', 'sanctity_sent'
        ]
        frameaxis_columns =basic_columns + [
            'bias_loyalty', 'bias_care', 'bias_fairness', 'bias_authority', 'bias_sanctity',
            'intensity_loyalty', 'intensity_care', 'intensity_fairness', 'intensity_authority', 'intensity_sanctity',
            'loyalty.virtue', 'loyalty.vice', 'care.virtue', 'care.vice',
            'fairness.vice', 'fairness.virtue', 'authority.virtue', 'authority.vice',
            'sanctity.vice', 'sanctity.virtue'
        ]

        # Create DataFrames for each file
        train_df = df[basic_columns]
        train_emfd_df = df[emfd_columns]
        train_frameaxis_df = df[frameaxis_columns]

        # Define paths for the new CSV files
        train_path = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/new/{target}_{mode}.csv'
        train_emfd_path = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/new/{target}_{mode}_emfd.csv'
        train_frameaxis_path = f'/home/qnnguyen/stance-detection/code/data/cb-dataset/plms/new/{target}_{mode}_frameaxis.csv'

        # Save the DataFrames to CSV files
        train_df.to_csv(train_path, index=False)
        train_emfd_df.to_csv(train_emfd_path, index=False)
        train_frameaxis_df.to_csv(train_frameaxis_path, index=False)

        # new_file_path
