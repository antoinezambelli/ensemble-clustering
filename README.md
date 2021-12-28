# ensemble-clustering

![Tests](https://github.com/antoinezambelli/ensemble-clustering/actions/workflows/unit-tests.yml/badge.svg) [![codecov](https://codecov.io/gh/antoinezambelli/ensemble-clustering/branch/main/graph/badge.svg?token=SPW6ID4OOM)](https://codecov.io/gh/antoinezambelli/ensemble-clustering)

Companion code to "Ensemble Method for Cluster Number Determination and Algorithm Selection in Unsupervised Learning" ([arXiv:2112.13680](https://arxiv.org/pdf/2112.13680.pdf)).

Builds an ensemble clustering framework, computes clusterings and computes metrics if given ground truth.

## Installation

This can be installed with `pip install git+[repo url]`.

### Python Packages
Contained in `setup.py` (note that it might be possible to use lower-version packages but this has not been tested):

- `fastcluster>=1.2.4`,
- `numpy>=1.20.3`,
- `scikit-learn>=1.0`,
- `scipy>=1.7.1`,
- `tqdm>=4.62.3`,

## Usage

Once the library is installed, it can be used as follows (this is also presented in the `examples/` directory).

### Parameters

The first step is to define parameters. 4 dictionaries are needed for full functionality.

- The metrics to use with each algorithm.
- Any higher-level parameters per algorithm.
- Hyperparameter ranges to tune over.
- `E` matrix construction and voting parameters.

```
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
        'init': ['k-means++'],
        'reassignment_ratio': np.geomspace(1e-4, 0.5, 2)
    },
}

# Define matrix parameters.
e_params = {
    'build': ['mode', 'raw'],
    'vote': ['row', 'col', 'full'],
    'ground_truth': 3
}
```

### Single Dataset

To cluster and analyze a single dataset, we can simply run (see `examples/single.py` for details)

```
from ensemble_clustering import Ensemble


X, y = make_blobs(n_samples=30000, centers=3, n_features=2, center_box=(-5, 5), random_state=1)

my_ensemble = Ensemble(algo_metrics, algo_params, h_params)
res, nc_res = my_ensemble(X, (2, 7), e_params)
```

### Multiple Datasets

To analyze multiple datasets (subsets for instance) we would run (see `examples/multi.py` for details)

```
from ensemble_clustering import Ensemble, E


full_res = []
my_ensemble = Ensemble(algo_metrics, algo_params, h_params)

for seed in range(1, 3):
    X, y = make_blobs(n_samples=30000, centers=3, n_features=2, center_box=(-5, 5), random_state=seed)
    
    # Get results.
    res, _ = my_ensemble(X, (2, 7))
    full_res.append(res)

my_E = E(full_res, my_ensemble.param_perms)
E_res = my_E(e_params)
```

### Paper Results

Code used to recreate the paper results can be found in `examples/paper.py`. Note that it will likely take several days to run depending on your hardware.

### Result Format

Results are presented in a detailed form to allow for multiple use-cases and to allow the user to aggregate as desired.

#### Ensemble

Calling the `Ensemble` class returns a tuple, where the second element is `None` unless E matrix parameters are specified. The first element follows this structure.

```
res = {
    'MiniBatchKMeans': [
        {'aic': 3, 'inertia': 3},
        {'aic': 3, 'inertia': 3}
    ],
    'linkage_vector': [
        {'inertia': 2, 'max_diff': 2},
        {'inertia': 3, 'max_diff': 3}
    ]
}
```

A dictionary with one key per algorithm, which has a list of every number of clusters result (by metric). The list is sorted in the same order as `my_ensemble.param_perms`.

The second tuple element, when not `None` is the output from calling `E` on a single dataset - which we discuss below.

#### Clustering

`Clustering` is intended to be an internal class not generally called explicitly. However, its output is the internal dicitionaries found in `res` above:

```
{'aic': 3, 'inertia': 3}
```

`Clustering` can be called to generate one-off results, for a given algorithm and hyperparameter combination.

#### E

The `E` class builds the E matrix and votes across the ensemble to obtain the number of clusters found in the dataset as well as the best algorithm-hyperparameter combination to use. The results follow this format

```
nc_res = {
    'mode': {
        'row': {'num_clusters': [2]},
        'col': {'num_clusters': [2]},
        'full': {'num_clusters': [3]}
    },
    'raw': {
        'row': {'num_clusters': [2]},
        'col': {'num_clusters': [2]},
        'full': {'num_clusters': [3]}
    },
    'best_algo': [
        [
            ('MiniBatchKMeans', {'init': 'k-means++', 'reassignment_ratio': 0.0001}),
            ('MiniBatchKMeans', {'init': 'k-means++', 'reassignment_ratio': 0.5}),
            ('linkage_vector', {'method': 'median', 'metric': 'euclidean'})
        ]
    ],
    'best_algo_global': [
        ('MiniBatchKMeans', {'init': 'k-means++', 'reassignment_ratio': 0.0001}),
        ('MiniBatchKMeans', {'init': 'k-means++', 'reassignment_ratio': 0.5}),
        ('linkage_vector', {'method': 'median', 'metric': 'euclidean'})
    ]
}
```

Where we have the following:

- The `mode`/`raw` and `row`/`col`/`full` keys are present if they were present in the e_params input when calling `E()`.
- The `num_clusters` lists contain one int per dataset passed in.
- The `best_algo_global` key contains a list of the top algorithm-hyperparameter combinations detected on all datasets.
- The `best_algo` list contains one list per dataset, with the top combinations for each dataset.

## Support

Functionality supported by the library.

### Algorithms

The following algorithms are currently supported.

- `GaussianMixture` (scikit-learn)
- `linkage_vector` (fastcluster)
- `MiniBatchKMeans` (scikit-learn)
- `SpectralClustering` (scikit-learn)

Additional algorithms can be added by modifying `clustering.py` and `utils.py` appropriately. Note, if they require additional processing (like `SpectralClustering` or `linkage_vector` did), then more work may be required.

We are planning to offer some limited support by adding requested algorithms as time permits (it's helpful if you could provide a high-level workflow using this algorithm: ie, for K-means we cluster N times and find the elbow in the curve). If contributors have modified the code and wish to open PRs please reach out in the issues.

In general, `clustering.py` and `utils.py` should be the only place where modifications are required. It should be relatively straightforward to add algorithms in the same way `linkage_vector` was added, by having a separate `get_[algo]_votes()` method. Then we simply need to ensure compatibility with metric functions.

In order to include an algorithm in the main `get_votes()` pipeline, we need to ensure that it follows the same logic (cluster N times, find the elbow) and has a compatible API (uses `fit()` then `predict()` or contains labels as an attribute after fit).

### Additional

Currently,

- `aic` and `bic` are only defined for `MiniBatchKMeans` and `GaussianMixture`.
- `inertia` and `silhouette_score` are defined for all algorithms.
- `linkage_vector` has additional `max_diff` and `elbow` metrics.

Additional metrics or other improvements can be addressed as time permits.
