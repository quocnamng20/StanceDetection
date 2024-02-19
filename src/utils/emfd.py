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

def interpret_moral_foundation(probability, sentiment, foundation, prob_thresholds, sent_thresholds):
    foundation_key = foundation.split('/')[0]  # E.g., 'care/harm' becomes 'care'

    prob_key = f'{foundation_key}_p'
    sent_key = f'{foundation_key}_sent'

    prob_thresh = prob_thresholds[prob_key]
    sent_thresh = sent_thresholds[sent_key]

    # Interpret probability score based on dynamic thresholds
    if 'moderate' in prob_thresh:
        level = "very strong" if probability > prob_thresh['high'] else \
                "strong" if probability > prob_thresh['moderate'] else \
                "moderate" if probability > prob_thresh['low'] else "weak"
    else:
        level = "strong" if probability > prob_thresh['high'] else "weak"

    # Interpret sentiment score based on dynamic thresholds
    if 'moderate' in sent_thresh:
        sentiment_desc = "support" if sentiment > sent_thresh['high'] else \
                         "against" if sentiment < sent_thresh['low'] else "neutral"
    else:
        sentiment_desc = "support" if sentiment > sent_thresh['high'] else \
                         "against" if sentiment < sent_thresh['low'] else "neutral"

    return f"{foundation.capitalize()} shows a {level} level of {sentiment_desc} expression."

def create_humanized_prompt_with_explanations(row, prob_thresholds, sent_thresholds, dataset, prompt_type):
    tweet = row['Tweet']
    target = row['Target']

    interpretations = [
        interpret_moral_foundation(row['care_p'], row['care_sent'], "care/harm", prob_thresholds, sent_thresholds),
        interpret_moral_foundation(row['fairness_p'], row['fairness_sent'], "fairness/cheating", prob_thresholds, sent_thresholds),
        interpret_moral_foundation(row['loyalty_p'], row['loyalty_sent'], "loyalty/betrayal", prob_thresholds, sent_thresholds),
        interpret_moral_foundation(row['authority_p'], row['authority_sent'], "authority/subversion", prob_thresholds, sent_thresholds),
        interpret_moral_foundation(row['sanctity_p'], row['sanctity_sent'], "sanctity/degradation", prob_thresholds, sent_thresholds),
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
    - Care/Harm: Focuses on empathy, nurturing, and kindness.
    - Fairness/Cheating: Revolves around justice, fairness, and equality.
    - Loyalty/Betrayal: Concerns group loyalty, trust, and patriotism.
    - Authority/Subversion: Involves respect for authority and tradition.
    - Sanctity/Degradation: Relates to purity, sanctity, and higher aspirations.

    Below is the Moral Foundations Interpretation towards the target "{target}" in the tweet:
    {combined_interpretation}
    Consider the text, subtext, regional and cultural references, and the moral foundation explanation and interpretation below to determine the stance expressed in this tweet towards the target "{target}". The possible stances are:
    {possible_stances}
    Tweet: {tweet}
    {question}
    """

    return prompt