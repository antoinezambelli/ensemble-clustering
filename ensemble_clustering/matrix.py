import itertools
import warnings
from typing import (
    Dict,
    List,
    Tuple
)

import numpy as np


class E():
    def __init__(self, meta_res: List, param_perms: Dict[str, List[Dict[Any]]]):
        '''
        On init, just build E according to mode or raw.
        '''
        self.meta_res = meta_res
        self.param_perms = param_perms

    def get_best_algo(self, ground_truth: int, single: bool) -> List[Tuple[str, Dict[Any]]]:
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
        ]  # Get top perms as per 'most matches across metrics' and 'most top performing perms out of all perms'.

        return top_perms

    def build_matrix(self, build: str):
        if build == 'mode':
            res = {k: list(np.argmax(np.bincount(list(sub_v.values()))) for sub_v in v) for k, v in self.res.items()}
        else:
            res = {k: list(val for sub_v in v for val in sub_v.values()) for k, v in self.res.items()}

        self.e = np.array(list(itertools.product(*res.values())))

        return self

    def get_num_clusters(self, vote: str) -> int:
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
        Loop over possible build/vote constructs and check performance.
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

    def __call__(self, e_params: Dict[Any]) -> Dict[Any]:
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
                self.build_matrix(e_params['build'])

                for vote in e_params['vote']:
                    num_clusters = self.get_num_clusters(vote)

                    E_res[build][vote]['num_clusters'].append(num_clusters)

        # Evaluation.
        if e_params.get('ground_truth', None) is not None:
            self.evaluate(E_res, e_params['ground_truth'])

        return E_res

