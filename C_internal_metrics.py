import numpy as np
import pandas as pd

def data_preprocessing(_data, clusters):
    # Assuming X, clusters is pandas DataFrame
    data = _data.copy()
    data = data.to_dict(orient="list")
    data = np.column_stack([data[col] for col in _data.columns])
    clusters = clusters.iloc[:, 0].values
    unique_labels, regularized_clusters = np.unique(clusters, return_inverse=True)
    regularized_clusters += 1
    return data, regularized_clusters

def compute_G_gc(cluster_data, total_objects_in_cluster):
    """
    Computes G_gc for a given cluster and variable.
    """
    G_gc = 0
    for variable_data in cluster_data.T:  # Iterate over variables (columns)
        unique_categories, counts = np.unique(variable_data, return_counts=True)
        proportions = counts / total_objects_in_cluster
        G_gc += 1 - np.sum(proportions**2)  # Calculate mutability
    return G_gc

def compute_WCM_k(data, clusters, k):
    """
    Computes WCM(k) based on the given clusters.
    """
    n, m = data.shape
    WCM_k = 0
    for g in range(1, k + 1):
        cluster_data = data[clusters == g]  # Filter data for the current cluster
        n_g = cluster_data.shape[0]
        if n_g == 0:
            continue  # Skip empty clusters
        G_gc_sum = compute_G_gc(cluster_data, n_g)
        WCM_k += (n_g / (n * m)) * G_gc_sum

    return WCM_k

def compute_PSFM_k(data, clusters, k):
    """
    Computes PSFM(k) for a specific value of k.
    """
    n, _ = data.shape
    data, regularized_clusters = data_preprocessing(data, clusters)
    clusters_1 = np.ones(n, dtype=int)

    WCM_1 = compute_WCM_k(data, clusters_1, 1)
    WCM_k = compute_WCM_k(data, regularized_clusters, k)
    if WCM_k == 0:  # Avoid division by zero
        PSFM_k = np.nan
    else:
        PSFM_k = ((n - k) * (WCM_1 - WCM_k)) / ((k - 1) * WCM_k)

    return PSFM_k

def compute_H_gc(cluster_data, total_objects_in_cluster):
    """
    Computes H_gc (entropy) for a given cluster and variable.
    """
    H_gc = 0
    for variable_data in cluster_data.T:  # Iterate over variables (columns)
        unique_categories, counts = np.unique(variable_data, return_counts=True)
        proportions = counts / total_objects_in_cluster
        H_gc += -np.sum(proportions * np.log2(proportions + 1e-10))  # Avoid log(0)
    return H_gc

def compute_WCE(data, clusters, k):
    """
    Computes WCE(k) based on the given clusters.
    """
    n, m = data.shape
    WCE_k = 0
    for g in range(1, k + 1):
        cluster_data = data[clusters == g]
        n_g = cluster_data.shape[0]
        if n_g == 0:
            continue  # Skip empty clusters
        H_gc_sum = compute_H_gc(cluster_data, n_g)
        WCE_k += (n_g / (n * m)) * H_gc_sum

    return WCE_k

def compute_PSFE_k(data, clusters, k):
    """
    Computes PSFE(k) for a specific value of k.
    """
    n, _ = data.shape
    data, regularized_clusters = data_preprocessing(data, clusters)
    clusters_1 = np.ones(n, dtype=int)

    WCE_1 = compute_WCE(data, clusters_1, 1)
    WCE_k = compute_WCE(data, regularized_clusters, k)

    if WCE_k == 0:  # Avoid division by zero
        PSFE_k = np.nan
    else:
        PSFE_k = ((n - k) * (WCE_1 - WCE_k)) / ((k - 1) * WCE_k)

    return PSFE_k

def computeBK_H_gc(cluster_data, total_objects_in_cluster):
    """
    Computes H_gc (entropy) for a given cluster and variable.
    """
    H_gc = []
    for variable_data in cluster_data.T:  # Iterate over variables (columns)
        unique_categories, counts = np.unique(variable_data, return_counts=True)
        proportions = counts / total_objects_in_cluster
        H_gc.append(-np.sum(proportions * np.log2(proportions + 1e-10)))
    return H_gc

def compute_H_E(data, clusters, k):
    """
    Computes H_E(k) based on the given clusters.
    """
    n = len(data)
    H_E_k = 0
    num_features = data.shape[1]

    unique_category_counts = np.array([len(np.unique(data[:, col])) for col in range(num_features)])

    for cluster_id in np.unique(clusters):
        cluster_indices = np.where(clusters == cluster_id)[0]
        cluster_data = data[cluster_indices]
        n_g = len(cluster_data)
        if n_g == 0:
            continue  # Skip empty clusters
        H_gc_sum = computeBK_H_gc(cluster_data, n_g)
        H_E_k += (n_g / n) * np.sum(np.array(H_gc_sum) / np.log2(unique_category_counts + 1e-10))

    return H_E_k

def compute_I_k(data, clusters_k, clusters_k_plus_1, k):
    """
    Computes I(k), the incremental expected entropy.
    """
    H_E_k = compute_H_E(data, clusters_k, k)
    H_E_k_plus_1 = compute_H_E(data, clusters_k_plus_1, k + 1)

    I_k_value = H_E_k - H_E_k_plus_1

    return I_k_value

def compute_BK_index(_data, clustering_algorithm, k):
    """
    Computes BK index for a specific value of k.
    """
    if k < 2:
        raise ValueError("k must be at least 2 to compute BK index.")

    clusters_k_minus_1 = clustering_algorithm(_data, k - 1) if k > 1 else np.ones(len(_data), dtype=int)
    clusters_k = clustering_algorithm(_data, k)
    clusters_k_plus_1 = clustering_algorithm(_data, k + 1)
    clusters_k_plus_2 = clustering_algorithm(_data, k + 2)

    data = _data.to_dict(orient="list")
    data = np.column_stack([data[col] for col in _data.columns])

    I_k_minus_1 = compute_I_k(data, clusters_k_minus_1, clusters_k, k - 1)
    I_k = compute_I_k(data, clusters_k, clusters_k_plus_1, k)
    I_k_plus_1 = compute_I_k(data, clusters_k_plus_1, clusters_k_plus_2, k + 1)

    BK_k = (I_k_minus_1 - I_k) - (I_k - I_k_plus_1)

    return BK_k
