'''
matrix.py: Contains E class which builds the E matrix and get num_clusters and best algorithm combinations.

Copyright(c) 2021, Antoine Emil Zambelli.
'''

import itertools
import warnings
from typing import (
    Dict,
    List,
    Optional,
    Tuple
)

import numpy as np


class E():
    def __init__(self, meta_res: List, param_perms: Dict[str, List[Dict]]):
        '''
        On init, just store inputs.
        Inputs:
            meta_res: list of outputs from Ensemble() calls, can be a single-element list.
            param_perms: see output from Ensemble.__init__(), dicitonary of hyperparams for each algo.
        Outputs:
            None.
        '''
        self.meta_res = meta_res
        self.param_perms = param_perms

    def get_best_algo(self, ground_truth: Optional[int], single: bool) -> List[Tuple[str, Dict]]:
        '''
        Finds the "best" algorithm-hyperparameter combination to use for a given dataset.
        Current logic is to pick combinations that match ground truth across the most metrics.
        Note that for a single dataset, this is likely to result in a tie.
        Additional logic included but not leveraged: 'most top performing perms out of all perms'.
        '''
        if ground_truth is None:
            return None

        if single:
            vote_vals = {
                k: np.sum(
                    [
                        [list(x.values()).count(ground_truth) / len(list(x.values())) for x in v]
                    ],
                    axis=0
                )
                for k, v in self.res.items()
            }  # Get percent of results that match ground truth for every hyperparam combination.
        else:
            vote_vals = {
                k: np.sum(
                    [
                        [list(x.values()).count(ground_truth) / len(list(x.values())) for x in v[k]]
                        for v in self.meta_res
                    ],
                    axis=0
                )
                for k, _ in self.meta_res[0].items()
            }  # Get percent of results that match ground truth for every hyperparam combination.

        idx_vals = {
            k: (
                np.max(v),  # Top performance.
                np.unique(np.where(v == np.max(v))[0]),  # Top performing perms.
                np.where(v == np.max(v))[0].shape[0] / len(self.param_perms[k])  # Stability of perms.
            )
            for k, v in vote_vals.items()
        }  # Get max match, location of max matches, and ratio of 'combinations that got max matches' to 'all combinations'.

        max_val = np.max([v[0] for k, v in idx_vals.items()])  # Absolute max count over all.
        max_perm = np.max([v[2] for k, v in idx_vals.items()])  # Absolute max combo performance over all.

        top_perms = [
            (k, self.param_perms[k][idx])
            for k, v in idx_vals.items()
            for idx in v[1]
            if v[0] == max_val
        ]  # Get top perms as per 'most matches across metrics'.

        return top_perms

    def build_matrix(self, build: str):
        '''
        Builds the E matrix according to 'mode' or 'raw' methods. 
        '''
        if build == 'mode':
            res = {k: list(np.argmax(np.bincount(list(sub_v.values()))) for sub_v in v) for k, v in self.res.items()}
        else:
            res = {k: list(val for sub_v in v for val in sub_v.values()) for k, v in self.res.items()}

        self.e = np.array(list(itertools.product(*res.values())))

        return self

    def get_num_clusters(self, vote: str) -> int:
        '''
        Get the estimated number of clusters from an E matrix.
        '''
        if vote == 'full':
            return np.argmax(np.bincount(self.e.flatten()))

        wombat = []  # Easter egg.
        e = self.e if vote == 'row' else self.e.T  # Row vs Col.
        for i in range(e.shape[0]):
            x = e[i, :].flatten()  # Elements belonging to that row (or algo if col).
            wombat.append(np.argmax(np.bincount(x.flatten())))

        return np.argmax(np.bincount(wombat))

    def evaluate(self, E_res, ground_truth: int):
        '''
        Loop over possible build/vote constructs and check/print performance.
        '''
        print('Ensemble Clustering metrics:')
        print('----------------------------')
        for build, votes in E_res.items():
            if 'best_algo' in build:
                continue

            for vote in votes:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=RuntimeWarning)
                    print(
                        '{}-{} --- Accuracy: {:.2f}%, Average Error: {:.2f}.'.format(
                            build,
                            vote,
                            100 * E_res[build][vote]['num_clusters'].count(ground_truth) / len(E_res[build][vote]['num_clusters']),
                            np.nanmean(
                                np.absolute(
                                    np.array([x for x in E_res[build][vote]['num_clusters'] if x != ground_truth]) - ground_truth
                                )
                            )
                        )
                    )

    def __call__(self, e_params: Dict) -> Dict:
        '''
        Call for each hyperparameter combination and algo (clustering algorithm).
        Inputs:
            e_params: E-matrix parameters with build and vote params.
        Outputs:
            E_res: "final" results for the workflow, containing estimated number of clusters and
                   best algorithm-hyperparameter combination to use per dataset and across all.
        '''
        E_res = {
            build: {
                vote: {
                    'num_clusters': []
                }
                for vote in e_params['vote']
            }
            for build in e_params['build']
        }  # Initialize results dictionary.
        E_res['best_algo'] = []  # Shared for all builds/votes.
        E_res['best_algo_global'] = self.get_best_algo(e_params.get('ground_truth', None), False)  # Overall best.

        # Loop over all data subsets.
        for res in self.meta_res:
            self.res = res
            E_res['best_algo'].append(
                self.get_best_algo(e_params.get('ground_truth', None), True)
            )  # Shared for all builds/votes.

            # Loop through all possible builds and votes.
            for build in e_params['build']:
                self.build_matrix(build)

                for vote in e_params['vote']:
                    num_clusters = self.get_num_clusters(vote)

                    E_res[build][vote]['num_clusters'].append(num_clusters)

        # Evaluation.
        if e_params.get('ground_truth', None) is not None:
            self.evaluate(E_res, e_params['ground_truth'])

        return E_res

