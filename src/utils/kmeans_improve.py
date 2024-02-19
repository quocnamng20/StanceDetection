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

def compute_improved_kmeans_thresholds(df, column_names):
    thresholds = {}
    for column in column_names:
        scaler = StandardScaler()
        values = scaler.fit_transform(df[column].values.reshape(-1, 1))

        best_silhouette = -1
        best_kmeans = None
        for n_clusters in range(2, 10):  # Experiment with range
            kmeans = KMeans(n_clusters=n_clusters, init='k-means++', n_init=10, random_state=0).fit(values)
            silhouette_avg = silhouette_score(values, kmeans.labels_)
            if silhouette_avg > best_silhouette:
                best_silhouette = silhouette_avg
                best_kmeans = kmeans

        centroids = np.sort(best_kmeans.cluster_centers_.ravel())

        # Set thresholds based on centroids
        if len(centroids) > 2:
            thresholds[column] = {'low': centroids[0], 'moderate': centroids[1], 'high': centroids[-1]}
        else:
            thresholds[column] = {'low': centroids[0], 'high': centroids[-1]}

    return thresholds