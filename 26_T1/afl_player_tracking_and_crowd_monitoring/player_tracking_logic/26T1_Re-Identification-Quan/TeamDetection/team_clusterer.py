# clustering/team_clusterer.py

import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

class TeamClusterer:
    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.dbscan = DBSCAN(eps=0.5, min_samples=3)
        self.scaler = StandardScaler()

    def remove_noise(self, embeddings):
        if len(embeddings) < 5:
            return embeddings, np.arange(len(embeddings))

        X = self.scaler.fit_transform(embeddings)
        labels = self.dbscan.fit_predict(X)

        valid_idx = np.where(labels != -1)[0]

        if len(valid_idx) < 3:
            return embeddings, np.arange(len(embeddings))

        return embeddings[valid_idx], valid_idx

    def cluster(self, embeddings):
        filtered_embeddings, valid_idx = self.remove_noise(embeddings)

        if len(filtered_embeddings) < self.n_clusters:
            return np.zeros(len(embeddings), dtype=int)

        cluster_labels = self.kmeans.fit_predict(filtered_embeddings)

        full_labels = np.full(len(embeddings), -1)
        full_labels[valid_idx] = cluster_labels

        return full_labels