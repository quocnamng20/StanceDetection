import pandas as pd
import numpy as np
import nltk
import json
import os
from nltk.tokenize import word_tokenize
from tqdm import tqdm

import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

def compute_kmeans_thresholds(df, column_names):
    thresholds = {}
    for column in column_names:
        # Scale the data
        scaler = StandardScaler()
        values = scaler.fit_transform(df[column].values.reshape(-1, 1))

        # Determine the optimal number of clusters using silhouette score
        silhouette_scores = []
        for n_clusters in range(2, 7):  # Example range, adjust as needed
            kmeans = KMeans(n_clusters=n_clusters, n_init='auto', random_state=0).fit(values)
            score = silhouette_score(values, kmeans.labels_)
            silhouette_scores.append((n_clusters, score))

        optimal_n_clusters = sorted(silhouette_scores, key=lambda x: x[1], reverse=True)[0][0]

        # Fit KMeans with optimal number of clusters
        kmeans = KMeans(n_clusters=optimal_n_clusters, n_init='auto', random_state=0).fit(values)
        centroids = np.sort(kmeans.cluster_centers_.ravel())

        # Adjust this part based on how you wish to interpret the clusters
        if len(centroids) > 2:
            thresholds[column] = {'low': centroids[0], 'moderate': centroids[1], 'high': centroids[-1]}
        else:  # Fallback if not enough clusters
            thresholds[column] = {'low': centroids[0], 'high': centroids[-1]}

    return thresholds