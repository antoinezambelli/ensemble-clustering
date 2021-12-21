import itertools

from tqdm import tqdm

from .clustering import Clustering
from .matrix import E


class Ensemble():
    def __init__(self, algo_metrics, algo_params, h_params):
        '''
        On init, just compute all the parameter combinations.
        '''
        # Store for later.
        self.algo_params = algo_params
        self.algo_metrics = algo_metrics

        # Define the parameter combinations for each algorithm.
        self.param_perms = {}
        for cluster_str, _ in algo_metrics.items():
            if cluster_str == 'SpectralClustering':
                # Compute perms for 'precomputed' case + other cases.
                self.param_perms[cluster_str] = [
                    dict(zip(h_params[cluster_str], v))
                    for v in itertools.product(
                        ['precomputed'], h_params[cluster_str]['metric'],
                        h_params[cluster_str]['n_neighbors'], [1.0]
                    )
                ] + [
                    dict(zip(h_params[cluster_str], v))
                    for v in itertools.product(
                        ['laplacian', 'rbf', 'sigmoid'], ['euclidian'],
                        [5], h_params[cluster_str]['gamma']
                    )
                ]
            else:
                self.param_perms[cluster_str] = [
                    dict(zip(h_params[cluster_str], v))
                    for v in itertools.product(*h_params[cluster_str].values())
                ]

    def generate_results(self, my_clust, algo_selections):
        res = {algo: [] for algo in algo_selections}  # Results holder.

        # Loop through all algorithms.
        for algo in tqdm(algo_selections, desc='Algorithms', ncols=100, leave=None):

            # Loop through all combinations for that algorithm.
            for h_perm in tqdm(self.param_perms[algo], desc=algo + ' combinations', ncols=100, leave=None):
                vote_dict = my_clust(h_perm, algo)
                res[algo].append(vote_dict)

        return res

    def __call__(self, X, k_range, e_params=None, algo_selections=None):
        '''
        On call, pass in workflow stuff (just algos to examine and (2,7) range for now) and datasets.
        '''
        nc = None
        my_clust = Clustering(X, k_range, self.algo_params, self.algo_metrics)  # Initialize clustering class.
        if algo_selections is None:
            algo_selections = [k for k, _ in self.algo_params.items()]  # Take all agorithms if none specified.

        res = self.generate_results(my_clust, algo_selections)  # Loop through all algorithms and combinations.

        # Build the E matrix and get vote results.
        if e_params is not None:
            my_E = E([res], self.param_perms)
            nc = my_E(e_params)

        return res, nc

