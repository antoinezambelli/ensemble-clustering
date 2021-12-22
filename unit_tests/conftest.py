'''
Fixtrues for unit tests.
'''

import pytest

import numpy as np
from sklearn.datasets import make_blobs


# Used to test full permutation logic.
@pytest.fixture
def algo_metrics_init():

    return {
        'MiniBatchKMeans': ['aic', 'bic', 'inertia', 'silhouette_score'],
        'SpectralClustering': ['inertia'],
        'GaussianMixture': ['bic'],
        'linkage_vector': ['elbow', 'inertia', 'max_diff']
    }

@pytest.fixture
def algo_params_init():

    return {
        'MiniBatchKMeans': {'n_clusters': None, 'compute_labels': True, 'random_state': 1},
        'SpectralClustering': {'n_clusters': None},
        'GaussianMixture': {'n_components': None},
        'linkage_vector': {}
    }

@pytest.fixture
def h_params_init():

    return {
        'GaussianMixture': {
            'covariance_type': ['spherical'],
            'reg_covar': np.geomspace(1e-8, 1e-2, 2)
        },
        'linkage_vector': {
            'method': ['centroid', 'median'],
            'metric': ['euclidean']
        },
        'MiniBatchKMeans': {
            'init': ['k-means++'],
            'reassignment_ratio': np.geomspace(1e-4, 0.5, 2)
        },
        'SpectralClustering': {
            'affinity': ['laplacian', 'precomputed'],
            'metric': ['l2'],  # Only used for 'precomputed'.
            'n_neighbors': [5],  # Only used for 'precomputed'.
            'gamma': [0.1, 1.0],  # Ignored for 'precomputed'.
        }
    }

@pytest.fixture
def algo_selections_run():

    return ['MiniBatchKMeans']

@pytest.fixture
def X_run():
    X, y = make_blobs(n_samples=10, centers=3, n_features=2, center_box=(-5, 5), random_state=1)

    return X

