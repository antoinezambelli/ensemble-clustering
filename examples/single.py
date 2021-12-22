from sklearn.datasets import make_blobs

from ensemble_clustering import Ensemble


# Set metrics to use with each algorithm.
algo_metrics = {
    'MiniBatchKMeans': ['aic', 'inertia'],
    'linkage_vector': ['inertia', 'max_diff']
}

# Set higher-level algorithm parameters. Set 'looping' param to None to auto-detect it.
algo_params = {
    'MiniBatchKMeans': {'n_clusters': None, 'compute_labels': True},
    'linkage_vector': {}
}

# Define hyperparameter ranges.
h_params = {
    'linkage_vector': {
        'method': ['centroid', 'median'],
        'metric': ['euclidean']
    },
    'MiniBatchKMeans': {
        'init': ['k-means++', 'random'],
        'reassignment_ratio': np.geomspace(1e-4, 0.5, 2)
    },
}

# Define matrix parameters.
e_params = {
    'build': ['mode', 'raw'],
    'vote': ['row', 'col', 'full'],
    'ground_truth': 3
}

X, y = make_blobs(n_samples=30000, centers=3, n_features=2, center_box=(-5, 5), random_state=1)

my_ensemble = Ensemble(algo_metrics, algo_params, h_params)
res, nc_res = my_ensemble(X, (2, 7), e_params)


