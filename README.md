# ensemble-clustering

![Tests](https://github.com/antoinezambelli/ensemble-clustering/actions/workflows/ci_workflow/badge.svg)

Companion code to "Ensemble Method for Cluster Number Determination and Algorithm Selection in Unsupervised Learning". TODO: link to paper. Builds an ensemble clustering framework, computes clusterings and validates results if given ground truth.

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

Once the library is installed, it can be used as follows (this is also presented in the `examples` directory).

### Parameters

The first step is to define parameters. 4 dictionaries are needed for full functionality.

- The metrics ot use with each algorithm.
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

TODO: describe raw res, nc_res and E_res output format in case more granularity is needed.

## Support

Functionality supported by the library.

### Algorithms

The following algorithms are currently supported.

- `GaussianMixture` (scikit-learn)
- `linkage_vector` (fastcluster)
- `MiniBatchKMeans` (scikit-learn)
- `SpectralClustering` (scikit-learn)

Additional algorithms can be added by modifying `clustering.py` appropriately. In the case where algorithms follow the same structure as GMM, K-means or Spectral then we can likely simply add an import statement. However, if they require additional processing (like Spectral did), then more work may be required.

We are planning to offer some limited support by adding requested algorithms as time permits (it's helpful if you could provide a high-level workflow using this algorithm: ie, for K-means we cluster N times and find the elbow in the curve). If contributors have modified the code and wish to open PRs please reach out in the issues.

TODO: detail procedure and gotchas.

### Additional

Additional metrics or other improvements can also be addressed as time permits.
