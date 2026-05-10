# utils/color_utils.py

import numpy as np
import cv2
from collections import Counter

def identify_referee(cluster_labels):
    counts = Counter(cluster_labels)
    counts.pop(-1, None)
    return min(counts, key=counts.get)


def extract_dominant_color(crop):
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.mean(hsv.reshape(-1, 3), axis=0)


def compute_cluster_colors(crops, labels):
    cluster_colors = {}

    for label in set(labels):
        if label == -1:
            continue

        colors = []
        for crop, l in zip(crops, labels):
            if l == label:
                colors.append(extract_dominant_color(crop))

        if colors:
            cluster_colors[label] = np.mean(colors, axis=0)

    return cluster_colors

def map_clusters_to_teams(cluster_colors):
    """
    Force clusters into:
    0 = team A
    1 = team B
    2 = referee
    """

    labels = list(cluster_colors.keys())
    colors = np.array(list(cluster_colors.values()))

    # Use HSV Hue (color identity)
    hues = colors[:, 0]

    # Sort clusters by hue
    sorted_idx = np.argsort(hues)

    mapping = {}

    # First two → teams
    mapping[labels[sorted_idx[0]]] = 0
    mapping[labels[sorted_idx[1]]] = 1

    # Last one → referee
    mapping[labels[sorted_idx[2]]] = 2

    return mapping