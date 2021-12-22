'''
Unit tests for utils.py
'''

import pytest

import numpy as np
from fastcluster import linkage_vector
from sklearn.cluster import MiniBatchKMeans
from sklearn.mixture import GaussianMixture

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

@pytest.mark.usefixtures('X_run')
@pytest.mark.usefixtures('param_perms_output')
def test_aic_bic(X_run, param_perms_output):
    gm = GaussianMixture(n_components=3, random_state=1, **param_perms_output['GaussianMixture'][0]).fit(X_run)
    labels = gm.predict(X_run)
    res_aic = aic(gm, X_run, 'GaussianMixture', labels)
    res_bic = bic(gm, X_run, 'GaussianMixture', labels)
    assert np.isclose(res_aic, 47.00303638131331)
    assert np.isclose(res_bic, 50.33147240424781)

    km = MiniBatchKMeans(
        n_clusters=3,
        random_state=1,
        compute_labels=True,
        **param_perms_output['MiniBatchKMeans'][0]
    ).fit(X_run)
    labels = km.labels_
    res_aic = aic(km, X_run, 'MiniBatchKMeans', labels)
    res_bic = bic(km, X_run, 'MiniBatchKMeans', labels)
    assert np.isclose(res_aic, 215.60766768110437)
    assert np.isclose(res_bic, 218.33093351805078)

@pytest.mark.usefixtures('X_run')
def test_inertia(X_run):
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

    res = inertia(X_run, labels)
    assert np.isclose(res, 87.20440924993774)
