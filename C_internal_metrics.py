import numpy as np
import pandas as pd

def compute_H_gc(cluster_data, total_objects_in_cluster):
    """
    Computes H_gc for a given cluster and variable.
    """
    H_gc = 0
    for variable_data in cluster_data.T:  # Iterate over variables (columns)
        unique_categories, counts = np.unique(variable_data, return_counts=True)
        proportions = counts / total_objects_in_cluster
        H_gc += -np.sum(proportions * np.log(proportions + 1e-9))  # Add small value to avoid log(0)
    return H_gc

def compute_H_E(data, clusters, k):
    """
    Computes H_E(k), the expected entropy of the dataset.
    """
    n, m = data.shape
    H_E_k = 0
    for g in range(1, k + 1):
        cluster_data = data[clusters == g]  # Filter data for the current cluster
        n_g = cluster_data.shape[0]
        if n_g == 0:
            continue  # Skip empty clusters
        H_gc_sum = compute_H_gc(cluster_data, n_g)
        H_E_k += (n_g / n) * (H_gc_sum / np.log(len(np.unique(data, axis=0))))
    return H_E_k

def compute_I_k(data, clusters, k):
    """
    Computes I(k), the incremental expected entropy.
    """
    H_E_k = compute_H_E(data, clusters, k)
    H_E_k_plus_1 = compute_H_E(data, clusters, k + 1)
    return H_E_k - H_E_k_plus_1

def compute_BK_index(data, clusters, k):
    """
    Computes the BK index for a specific value of k.
    """
    I_k_minus_1 = compute_I_k(data, clusters, k - 1)
    I_k = compute_I_k(data, clusters, k)
    I_k_plus_1 = compute_I_k(data, clusters, k + 1)
    return (I_k_minus_1 - I_k) - (I_k - I_k_plus_1)

#>>>>>>>>>>>>>>>>>>>>>>>>
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
    WCM_1 = compute_WCM_k(data, clusters, 1)
    WCM_k = compute_WCM_k(data, clusters, k)
    if WCM_k == 0:  # Avoid division by zero
        PSFM_k = np.nan  # Use NaN for undefined PSFM
    else:
        PSFM_k = ((n - k) * (WCM_1 - WCM_k)) / ((k - 1) * WCM_k)
    
    return PSFM_k

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def compute_H_gc(cluster_data, total_objects_in_cluster):
    """
    Computes H_gc (entropy) for a given cluster and variable.
    """
    H_gc = 0
    for variable_data in cluster_data.T:  # Iterate over variables (columns)
        unique_categories, counts = np.unique(variable_data, return_counts=True)
        proportions = counts / total_objects_in_cluster
        # Avoid log(0) and compute entropy
        H_gc += -np.sum(proportions * np.log(proportions + 1e-10))
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
        
        # Compute H_gc for the current cluster
        H_gc_sum = compute_H_gc(cluster_data, n_g)
        WCE_k += (n_g / (n * m)) * H_gc_sum  # Accumulate contributions to WCE(k)
    
    return WCE_k

def compute_PSFE_k(data, clusters, k):
    """
    Computes PSFE(k) for a specific value of k.
    """
    n, _ = data.shape
    WCE_1 = compute_WCE(data, clusters, 1)
    WCE_k = compute_WCE(data, clusters, k)
    
    if WCE_k == 0:  # Avoid division by zero
        PSFE_k = np.nan  # Use NaN for undefined PSFE
    else:
        PSFE_k = ((n - k) * (WCE_1 - WCE_k)) / ((k - 1) * WCE_k)
    
    return PSFE_k
