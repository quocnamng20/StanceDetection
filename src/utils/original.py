import replicate

import os
import json
import nltk
import numpy as np
import pandas as pd
from nltk.tokenize import word_tokenize
from tqdm import tqdm


import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('/home/qnnguyen/stance-detection/code')

from src.utils.kmeans import compute_kmeans_thresholds
from src.utils.kmeans_improve import compute_improved_kmeans_thresholds

def create_humanized_prompt_with_explanations(row, dataset, prompt_type):
    tweet = row['Tweet']
    target = row['Target']
    if prompt_type == 'few-shot':
        tweet_detailed = ""
        question = f"""Tweet: {tweet}"""
    else:    
        if dataset == 'semeval':
                tweet_detailed = f"""
        Consider the text, subtext, regional and cultural references to determine the stance expressed in this tweet towards the target "{target}". The possible stances are:
        - support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
        - against: The tweet opposes or criticizes the target, either explicitly or implicitly.
        - none: The tweet is neutral, doesn’t have a stance towards the target or unrelated to the target.
        """
                question = f"""
        Tweet: {tweet}
        Based on the tweet, what is the stance towards the target "{target}"? Please only respond one word: 'support', 'against', 'none'. Do NOT add any additional text."""
        
        else:
                tweet_detailed = f"""
        Consider the text, subtext, regional and cultural references to determine the stance expressed in this tweet towards the target "{target}". The possible stances are:
        - support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.
        - against: The tweet opposes or criticizes the target, either explicitly or implicitly.
        """
                question = f"""
        Tweet: {tweet}
        Based on the tweet, what is the stance towards the target "{target}"? Please only respond one word: 'support' or 'against'. Do NOT add any additional text."""

    return f"""{tweet_detailed}{question}"""
