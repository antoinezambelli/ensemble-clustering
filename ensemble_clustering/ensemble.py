import itertools
from typing import (
    Dict,
    List,
    Optional,
    Tuple,
    Union
)

import numpy as np
from tqdm import tqdm

from .clustering import Clustering
from .matrix import E


class Ensemble():
    def __init__(self, algo_metrics: Dict[str, List[str]], algo_params: Dict[str, Dict], h_params: Dict[str, Dict]):
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
                        [x for x in h_params[cluster_str]['affinity'] if x != 'precomputed'], ['euclidian'],
                        [5], h_params[cluster_str]['gamma']
                    )
                ]
            else:
                self.param_perms[cluster_str] = [
                    dict(zip(h_params[cluster_str], v))
                    for v in itertools.product(*h_params[cluster_str].values())
                ]

    def generate_results(self, my_clust: Clustering, algo_selections: List[str]) -> Dict[str, List[Dict]]:
        res = {algo: [] for algo in algo_selections}  # Results holder.

        # Loop through all algorithms.
        for algo in tqdm(algo_selections, desc='Algorithms', ncols=100, leave=None):

            # Loop through all combinations for that algorithm.
            for h_perm in tqdm(self.param_perms[algo], desc=algo + ' combinations', ncols=100, leave=None):
                vote_dict = my_clust(h_perm, algo)
                res[algo].append(vote_dict)

        return res

    def __call__(
            self,
            X,
            k_range: Union[List[int], Tuple[int, int]],
            e_params: Optional[Dict]=None,
            algo_selections: Optional[List[str]]=None
        ) -> Union[Tuple[Dict[str, List[Dict]], Dict], Tuple[Dict[str, List[Dict]], None]]:
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

