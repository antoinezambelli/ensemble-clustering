'''
clustering.py: Contains Clustering class which clusters for a given algo-param combination, for all its metrics.

Copyright(c) 2021, Antoine Emil Zambelli.
'''

from inspect import signature
from typing import (
    Dict,
    List,
    Tuple,
    Union
)

from fastcluster import linkage_vector
from sklearn.cluster import SpectralClustering, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

from .utils import (
    aic,
    bic,
    elbow,
    get_n,
    hca_metrics,
    inertia,
    max_diff
)


class Clustering():
    def __init__(
            self,
            X,
            k_range: Union[List[int], Tuple[int, int]],
            algo_params: Dict[str, Dict],
            algo_metrics: Dict[str, List[str]]
        ):
        '''
        On init, just store values.
        Inputs:
            X: dataset to cluster.
            k_range: range to attempt for n_cluster trials, ie, (2, 7).
            algo_params: dicitonary of "higher-level" params to pass in to algorithms.
            algo_metrics: dictionary of metrics to use with each algorithm.
        Outputs:
            None.
        '''
        self.X = X
        self.k_range = k_range
        self.algo_params = algo_params
        self.algo_metrics = algo_metrics

    def get_hca_votes(self, h_params: Dict, algo: str) -> Dict[str, int]:
        '''
        linkage_vector-specific code withe the same functionality as get_votes(). Builds dendrogram and
        applies metrics to it to find number of clusters.
        '''
        for _ in tqdm(range(1), desc=algo, ncols=100, leave=None):
            c_params = {k: (v if v else e) for k, v in self.algo_params[algo].items()}
            c_params.update(h_params)

            Z = globals()[algo](self.X, **c_params)  # Access the appropriate model object.

        vote_dict = {}
        for m_str in self.algo_metrics[algo]:

            # Cases: inertia/silhouette (shared methods require hca() call); elbow/max_diff (hca-specific methods).
            if m_str in ['inertia', 'silhouette_score']:
                vote_dict[m_str] = hca_metrics(self.X, self.k_range, Z, m_str)
            else:
                vote_dict[m_str] = globals().get(m_str, None)(Z)

        return vote_dict

    def run_trial(self, graph, h_params: Dict, n_c: int, algo: str):
        '''
        Generate labels for that cluster number trial.
        '''
        c_params = {k: (v if v else n_c) for k, v in self.algo_params[algo].items()}
        h_params = {k: v for k,v in h_params.items() if k not in ['metric', 'n_neighbors']}
        c_params.update(h_params)

        model = globals()[algo](**c_params).fit(graph)  # Access the appropriate model object.
        labels = model.labels_ if hasattr(model, 'labels_') else model.predict(graph)

        return model, labels

    def compute_graph(self, h_params: Dict):
        '''
        Compute nearest-neighbors graph if needed. Currently only used for SpectralClustering.
        '''
        try:
            nbrs = NearestNeighbors(
                n_neighbors=h_params['n_neighbors'],
                algorithm='auto',
                metric=h_params['metric']
            ).fit(self.X)
        except ValueError as err:
            nbrs = NearestNeighbors(n_neighbors=h_params['n_neighbors'], algorithm='auto').fit(self.X)
        graph = nbrs.kneighbors_graph(self.X, mode='distance')

        return graph

    def get_votes(self, h_params: Dict, algo: str) -> Dict[str, int]:
        '''
        Loop through possible num clusters, use metrics to build elbow curves, find the elbow in each curve.
        '''
        res = {m_str: [] for m_str in self.algo_metrics[algo]}  # Storage for metric values.
        graph = self.X  # Naming for compatibility.

        # Handle Spectral case that requires an actual graph.
        if algo == 'SpectralClustering' and h_params.get('affinity', None) == 'precomputed':
            graph = self.compute_graph(h_params)

        # Loop over possible number of clusters and store results for each metric.
        for n_c in tqdm(range(self.k_range[0], self.k_range[1]), desc='n clusters', ncols=100, leave=None):

            model, labels = self.run_trial(graph, h_params, n_c, algo)  # Get clustering for current n_c.

            for m_str in self.algo_metrics[algo]:
                metric = globals().get(m_str, None)  # Use global function if available.

                if not metric:
                    metric = getattr(model, m_str)  # Get attribute - could be a model class method.

                # Call metric function.
                res[m_str].append(
                    metric(model, self.X, algo, labels)
                    if globals().get(m_str, None) and len(signature(globals().get(m_str, None)).parameters) == 4
                    else metric(self.X, labels)
                )

        vote_dict = {k: get_n(v) + 2 for k, v in res.items()}  # Note indexing offset.

        return vote_dict

    def __call__(self, h_params: Dict, algo: str) -> Dict[str, int]:
        '''
        Call for each hyperparameter combination and algo (clustering algorithm).
        Inputs:
            h_params: hyperparameters to look at - one element of Ensemble().param_perms[algo].
            algo: algorithm to use.
        Outputs:
            A "vote_dict" over metrics, ie, {'aic': 3, 'inertia': 3}.
        '''

        # Handle HCA separately, as it requires a different code structure.
        if algo == 'linkage_vector':
            return self.get_hca_votes(h_params, algo)

        return self.get_votes(h_params, algo)

