'''
paper.py: Example of how to call the workflow to reproduce results from
"Ensemble Method for Cluster Number Determination and Algorithm Selection in Unsupervised Learning".

Copyright(c) 2021, Antoine Emil Zambelli.
'''

from sklearn.datasets import make_blobs
from tqdm import tqdm

from ensemble_clustering import Ensemble, E


# Set metrics to use with each algorithm.
algo_metrics = {
    'MiniBatchKMeans': ['aic', 'bic', 'inertia', 'silhouette_score'],
    'SpectralClustering': ['inertia', 'silhouette_score'],
    'GaussianMixture': ['aic', 'bic', 'inertia', 'silhouette_score'],
    'linkage_vector': ['elbow', 'inertia', 'silhouette_score', 'max_diff']
}

# Set higher-level algorithm parameters. Set 'looping' param to None to auto-detect it.
algo_params = {
    'MiniBatchKMeans': {'n_clusters': None, 'compute_labels': True},
    'SpectralClustering': {'n_clusters': None},
    'GaussianMixture': {'n_components': None},
    'linkage_vector': {}
}

# Define hyperparameter ranges.
h_params = {
    'GaussianMixture': {
        'covariance_type': ['diag', 'tied', 'spherical'],
        'reg_covar': np.geomspace(1e-8, 1e-2, 6)
    },
    'linkage_vector': {
        'method': ['centroid', 'median', 'single', 'ward'],
        'metric': ['euclidean']
    },
    'MiniBatchKMeans': {
        'init': ['k-means++', 'random'],
        'reassignment_ratio': np.geomspace(1e-4, 0.5, 8)
    },
    'SpectralClustering': {
        'affinity': ['laplacian', 'precomputed', 'rbf', 'sigmoid',],
        'metric': ['cosine', 'l2', 'l1'],  # Only used for 'precomputed'.
        'n_neighbors': [5, 20, 100],  # Only used for 'precomputed'.
        'gamma': [0.1, 1.0, 10.0],  # Ignored for 'precomputed'.
    }
}

# Define matrix parameters.
e_params = {
    'build': ['mode', 'raw'],
    'vote': ['row', 'col', 'full'],
    'ground_truth': 3
}

full_res = []
my_ensemble = Ensemble(algo_metrics, algo_params, h_params)

for seed in tqdm(range(1, 101), desc='Datasets', ncols=100):
    X, y = make_blobs(n_samples=30000, centers=3, n_features=2, center_box=(-5, 5), random_state=seed)
    
    # Get results.
    res, _ = my_ensemble(X, (2, 7))
    full_res.append(res)

my_E = E(full_res, my_ensemble.param_perms)
E_res = my_E(e_params)
