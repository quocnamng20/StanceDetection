import replicate

import os
import json
import nltk
import sys
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
sys.path.append('/home/qnnguyen/stance-detection/code')

from src.utils.kmeans import compute_kmeans_thresholds
from src.utils.kmeans_improve import compute_improved_kmeans_thresholds

def interpret_moral_foundation_combined(bias, intensity, virtue, vice, foundation, target, bias_thresholds, intensity_thresholds):
    bias_key = f'bias_{foundation}'
    intensity_key = f'intensity_{foundation}'

    bias_thresh = bias_thresholds[bias_key]
    intensity_thresh = intensity_thresholds[intensity_key]

    # Adjust interpretation logic to handle potential absence of 'moderate' key
    if 'moderate' in bias_thresh:
        bias_desc = "a slight positive" if bias > bias_thresh['high'] else \
                    "a slight negative" if bias < bias_thresh['low'] else "a neutral"
    else:
        bias_desc = "positive" if bias > bias_thresh['high'] else "negative"

    if 'moderate' in intensity_thresh:
        intensity_desc = "moderate" if intensity > intensity_thresh['moderate'] else "low"
    else:
        intensity_desc = "high" if intensity > intensity_thresh['high'] else "low"

    aspect = "virtue" if virtue > vice else "vice"

    return f"{foundation.capitalize()} towards {target} shows {bias_desc} bias with {intensity_desc} intensity, predominantly as a {aspect}."


def create_humanized_prompt_with_explanations(row, bias_thresholds, intensity_thresholds, dataset, prompt_type):
    tweet = row['Tweet']
    target = row['Target']

    interpretations = [
        interpret_moral_foundation_combined(row['bias_loyalty'], row['intensity_loyalty'], row['loyalty.virtue'], row['loyalty.vice'], "loyalty", target, bias_thresholds, intensity_thresholds),
        interpret_moral_foundation_combined(row['bias_care'], row['intensity_care'], row['care.virtue'], row['care.vice'], "care", target, bias_thresholds, intensity_thresholds),
        interpret_moral_foundation_combined(row['bias_fairness'], row['intensity_fairness'], row['fairness.virtue'], row['fairness.vice'], "fairness", target, bias_thresholds, intensity_thresholds),
        interpret_moral_foundation_combined(row['bias_authority'], row['intensity_authority'], row['authority.virtue'], row['authority.vice'], "authority", target, bias_thresholds, intensity_thresholds),
        interpret_moral_foundation_combined(row['bias_sanctity'], row['intensity_sanctity'], row['sanctity.virtue'], row['sanctity.vice'], "sanctity", target, bias_thresholds, intensity_thresholds),
    ]
    combined_interpretation = " ".join(interpretations)
    if prompt_type == 'few-shot':
        possible_stances = ""
        question = ""
    else:    
        if dataset == 'semeval':
            possible_stances = "- support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.\n- against: The tweet opposes or criticizes the target, either explicitly or implicitly.\n- none: The tweet is neutral, doesn’t have a stance towards the target or unrelated to the target."
            
            question = f"""Based on the tweet and the morality foundation explanation and interpretation, what is the stance towards the target "{target}"? Please only respond one word: 'support', 'against', 'none'."""
        else:
            possible_stances = "- support: The tweet has a positive or supportive attitude towards the target, either explicitly or implicitly.\n- against: The tweet opposes or criticizes the target, either explicitly or implicitly."
            question = f"""Based on the tweet and the morality foundation explanation and interpretation, what is the stance towards the target "{target}"? Please only respond one word: 'support' or 'against'."""
    if prompt_type == 'few-shot':
        prompt = f"""
        Tweet: {tweet}
        The Moral Foundations Interpretation towards the target "{target} are:
        {combined_interpretation}"""
    else:
        prompt = f"""
Moral Foundations Explanation:
- Loyalty: Pertains to allegiance to a group or community.
- Care: Concerns empathy and protecting others from harm.
- Fairness: Involves ideas of justice, rights, and equality.
- Authority: Relates to respect for tradition and legitimate authority.
- Sanctity: Revolves around the concept of purity or disgust.

Below is the Morality Foundations Interpretation towards the target "{target}" in the tweet:
{combined_interpretation}
Consider the text, subtext, regional and cultural references, and the morality foundation explanation and interpretation below to determine the stance expressed in this tweet towards the target "{target}". The possible stances are:
{possible_stances}
Tweet: {tweet}
{question}
"""
    return prompt