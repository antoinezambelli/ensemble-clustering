import numpy as np
from scipy.cluster.hierarchy import fcluster
from sklearn.metrics import silhouette_score


def get_n(Y_p: np.ndarray) -> int:
    '''
    Convenience method: finds elbow in curve by using vector rejection.
    '''
    max_idx = np.argmax(Y_p)

    Y_p = Y_p[max_idx:]  # truncate data from max (not origin) to endpoint.
    b = np.array([len(Y_p), Y_p[-1] - Y_p[0]])  # Vector from origin to end.
    norm_vec = [0]  # Initial point ignored.

    for i in range(1, len(Y_p)):
        p = np.array([i, Y_p[i] - Y_p[0]])  # Vector from origin to current point on curve.
        d = np.linalg.norm(p - (np.dot(p, b) / np.dot(b, b)) * b)  # Distance from point to b.

        norm_vec.append(d)

    # Pick the longest connecting line - note max_idx added to slice back into original data.
    return max_idx + np.argmax(norm_vec)

def max_diff(Z) -> int:
    return len(Z) - np.argmax(np.diff(Z[:,2]))

def elbow(Z) -> int:
    return len(Z) - (np.argmax(np.diff(Z[:,2], 2)) + 1)

def hca_metrics(X, k_range, Z, sub_str: str) -> int:
    # Loop down the dendrogram and try different cutoffs - choose by sub_string method.
    rr = []
    for i in range(k_range[0], k_range[1]):
        max_d = (Z[-i, 2] + Z[-i + 1, 2]) / 2.0  # Midpoint between joins.
        labels = fcluster(Z, max_d, criterion='distance')  # Labels with that candidate cutoff.

        rr.append(globals().get(sub_str, None)(X, labels))  # Inertia/silhouette of that cutoff.

    # Find cutoff with best inertia/silhouette 'elbow' - note + 2 to offset loop index.
    max_d = (Z[-(get_n(rr) + 2), 2] + Z[-(get_n(rr) + 2) + 1, 2]) / 2.0
    labels = fcluster(Z, max_d, criterion='distance')

    return len(np.unique(labels))

def aic(model, X, algo: str, labels):
    if algo == 'GaussianMixture':
        n_params = model._n_parameters()
    elif algo == 'MiniBatchKMeans':
        # Ref: https://stats.stackexchange.com/questions/85929/corrected-aic-aicc-for-k-means?rq=1.
        n_params = len(np.unique(labels)) * (X.shape[1] + 1)

    return -2 * model.score(X) * X.shape[0] + 2 * n_params

def bic(model, X, algo: str, labels):
    if algo == 'GaussianMixture':
        n_params = model._n_parameters()
    elif algo == 'MiniBatchKMeans':
        # Ref: https://stats.stackexchange.com/questions/85929/corrected-aic-aicc-for-k-means?rq=1.
        n_params = len(np.unique(labels)) * (X.shape[1] + 1)

    return -2 * model.score(X) * X.shape[0] + n_params * np.log(X.shape[0])

def inertia(X, labels):
    # Manual inertia function...sum of squared distances between point and it's assigned center.
    labs = np.unique(labels)
    centroids = np.zeros_like(X)  # To store centroids for easier computation.

    # Compute centroid locations.
    for unique_lab in labs:
        lab_idx = np.where(labels == unique_lab)  # Index of points of a specific label.

        pts = X[lab_idx]
        centroids[lab_idx] = np.mean(pts, axis=0)  # Assign centroid value to correct indices.

    # Compute sum of squared distances.
    return np.sum((X - centroids) ** 2)

