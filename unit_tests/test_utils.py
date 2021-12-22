'''
Unit tests for utils.py
'''

import pytest

import numpy as np
from fastcluster import linkage_vector

from ensemble_clustering.utils import (
    aic,
    bic,
    elbow,
    get_n,
    hca_metrics,
    inertia,
    max_diff
)


def test_get_n():
    a = [15, 20, 5, 4, 3, 2, 1]
    z = get_n(a)
    assert z == 2

def test_max_diff():
    a = np.array(
        [
            [1, 1, 10],
            [1, 1, 7],
            [1, 1, 6],
        ]
    )
    z = max_diff(a)
    assert z == 2

def test_elbow():
    a = np.array(
        [
            [1, 1, 10],
            [1, 1, 7],
            [1, 1, 6],
        ]
    )
    z = max_diff(a)

    assert z == 2

@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('X_run')
def test_hca_metrics(param_perms_output, X_run):
    Z = globals()['linkage_vector'](X_run, **param_perms_output['linkage_vector'][0])

    res = hca_metrics(X_run, (2, 7), Z, 'inertia')
    assert res == 5

    res = hca_metrics(X_run, (2, 7), Z, 'silhouette_score')
    assert res == 5

# TODO: need param perms and algo_params_init.
# def test_aic():
#     # Init a gaussian, a kmeans. Fit to X and run aic.

#     assert True
