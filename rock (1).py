import pandas as pd
import numpy as np
import heapq
import warnings
import tqdm
from numba import jit
from scipy.sparse import csr_matrix

# Import the Cython functions
from rock_cython import compute_adjacency_matrix_cython, label_remaining_points_cython, jaccard_similarity

warnings.simplefilter(action='ignore', category=FutureWarning)

def preprocess_data(df):
    """Convert DataFrame to a format suitable for the ROCK algorithm."""
    df_copy = df.copy()

    # Preprocess the DataFrame
    for colnum in range(len(df_copy.columns)):
        df_copy.iloc[:, colnum] = df_copy.iloc[:, colnum].astype(str)
        df_copy.iloc[:, colnum] = df_copy.iloc[:, colnum].apply(lambda x: f"{colnum}|{x}")

    return df_copy.to_numpy()

# def jaccard_similarity(set1, set2):
#     """Calculate the Jaccard similarity between two sets."""
#     intersection = len(set1.intersection(set2))
#     union = len(set1.union(set2))
#     return intersection / union if union != 0 else 0

def calculate_links(data, threshold):
    """Calculate the links between pairs of points."""
    adjacency_matrix = compute_adjacency_matrix_cython(data, threshold)  # Use Cython function
    return calculate_links_numba(adjacency_matrix)

@jit(nopython=True)
def calculate_links_numba(adjacency_matrix):
    """Calculate the links between pairs of points using matrix multiplication with Numba."""
    n = adjacency_matrix.shape[0]
    links_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                links_matrix[i, j] = np.sum(adjacency_matrix[i] * adjacency_matrix[j])
    return links_matrix


def calculate_goodness_measure(Ci, Cj, links_matrix, threshold):
    """Calculate the goodness measure between two clusters Ci and Cj."""
    ni = len(Ci)  # Size of cluster Ci
    nj = len(Cj)  # Size of cluster Cj
    link_Ci_Cj = sum(links_matrix[i, j] for i in Ci for j in Cj)  # Cross links
    z = 1 + 2 * (1 - threshold / (1 + threshold))
    goodness = link_Ci_Cj / ((ni + nj) ** z - ni ** z - nj ** z)
    return goodness

def ROCK(df_original, sample_size=30, k=2, threshold=0.2, representativeness_fraction=0.5, min_cluster_size_percent=0.1, MIN_NEIGHBORS=2, batch_size=100):
    """ROCK Clustering Algorithm with one consolidated function and Numba optimization."""

    data = preprocess_data(df_original)  # Preprocess the DataFrame

    sampled_points = data[np.random.choice(data.shape[0], sample_size, replace=False), :]

    links_matrix = calculate_links(sampled_points, threshold)

    local_heaps = {i: [] for i in range(sampled_points.shape[0])}
    clusters = {i: [i] for i in range(sampled_points.shape[0])}

    for i in range(sampled_points.shape[0]):
        for j in range(sampled_points.shape[0]):
            if links_matrix[i, j] > 0 and i != j:
                goodness = calculate_goodness_measure(clusters[i], clusters[j], links_matrix, threshold)
                heapq.heappush(local_heaps[i], (-goodness, j))

    global_heap = []
    for i in clusters.keys():
        if local_heaps[i]:
            max_goodness, max_index = local_heaps[i][0]
            heapq.heappush(global_heap, (max_goodness, i, max_index))
        else:
            heapq.heappush(global_heap, (0, i, -1))

    while len(clusters) > k:
        if not global_heap:
            break

        max_goodness, u, v = heapq.heappop(global_heap)

        if u not in clusters or v not in clusters:
            continue

        w = max(clusters.keys()) + 1
        clusters[w] = clusters[u] + clusters[v]
        del clusters[u], clusters[v]

        links_matrix = np.pad(links_matrix, ((0, 1), (0, 1)), mode='constant')
        for x in list(clusters.keys()):
            if x != w:
                links_matrix[w, x] = links_matrix[u, x] + links_matrix[v, x]
                links_matrix[x, w] = links_matrix[w, x]

                goodness = calculate_goodness_measure(clusters[x], clusters[w], links_matrix, threshold)
                heapq.heappush(local_heaps[x], (-goodness, w))

        local_heaps[w] = []
        for x in list(clusters.keys()):
            if x != w:
                goodness = calculate_goodness_measure(clusters[w], clusters[x], links_matrix, threshold)
                heapq.heappush(local_heaps[w], (-goodness, x))

        global_heap = []
        for i in clusters.keys():
            if local_heaps[i]:
                max_goodness, max_index = local_heaps[i][0]
                heapq.heappush(global_heap, (max_goodness, i, max_index))

    print("\nRunning final labeling on the entire dataset...\n")

    # Process the data in batches
    final_labels = np.zeros(len(data), dtype=int)
    for start in range(0, len(data), batch_size):
        end = min(start + batch_size, len(data))
        batch_data = data[start:end]
        batch_labels = label_remaining_points_cython(batch_data, sampled_points, clusters, threshold,
                                              representativeness_fraction, min_cluster_size_percent, MIN_NEIGHBORS)
        final_labels[start:end] = batch_labels

    return final_labels
