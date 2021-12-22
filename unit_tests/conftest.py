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
def param_perms_output():

    return {
        'MiniBatchKMeans': [
            {'init': 'k-means++', 'reassignment_ratio': 0.0001},
            {'init': 'k-means++', 'reassignment_ratio': 0.5}
        ],
        'SpectralClustering': [
            {'affinity': 'precomputed', 'metric': 'l2', 'n_neighbors': 5, 'gamma': 1.0},
            {'affinity': 'laplacian', 'metric': 'euclidian', 'n_neighbors': 5, 'gamma': 0.1},
            {'affinity': 'laplacian', 'metric': 'euclidian', 'n_neighbors': 5, 'gamma': 1.0}
        ],
        'GaussianMixture': [
            {'covariance_type': 'spherical', 'reg_covar': 1e-08},
            {'covariance_type': 'spherical', 'reg_covar': 0.01}
        ],
        'linkage_vector': [
            {'method': 'centroid', 'metric': 'euclidean'},
            {'method': 'median', 'metric': 'euclidean'}
        ]
    }

@pytest.fixture
def algo_selections_run():

    return ['MiniBatchKMeans']

@pytest.fixture
def X_run():
    X, y = make_blobs(n_samples=10, centers=3, n_features=2, center_box=(-5, 5), random_state=1)

    return X

@pytest.fixture
def graph_output():
    graph = np.array(
        [[0., 2.22539366, 1.47635992, 1.88472918, 0., 0., 0., 0., 0., 0.93293083],
        [0., 0., 1.27350236, 0.78148202, 0., 0., 0., 0.89435066, 0., 1.55652124],
        [1.47635992, 1.27350236, 0., 0.56089504, 0., 0., 0., 0., 0., 0.54500526],
        [0., 0.78148202, 0.56089504, 0., 0., 0., 0., 1.62262196, 0., 1.01483258],
        [0., 0., 0., 6.57778932, 0., 1.77368928, 1.51486116, 0., 2.29426447, 0.],
        [0., 0., 4.89500851, 0., 1.77368928, 0., 2.12365567, 0., 1.85780729, 0.],
        [0., 0., 0., 6.04623555, 1.51486116, 2.12365567, 0., 0., 1.18155842, 0.],
        [0., 0.89435066, 2.15327146, 1.62262196, 0., 0., 0., 0., 0., 2.44822292],
        [0., 0., 0., 4.89726912, 2.29426447, 1.85780729, 1.18155842, 0., 0., 0.],
        [0.93293083, 1.55652124, 0.54500526, 1.01483258, 0., 0., 0., 0., 0., 0.]]
    )

    return graph

@pytest.fixture
def trial_output_labels():

    return np.array([2, 0, 2, 2, 1, 4, 1, 0, 3, 2])
