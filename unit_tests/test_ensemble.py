'''
Unit tests for ensemble.py
'''

import pytest

from ensemble_clustering import Ensemble, Clustering


@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('h_params_init')
@pytest.mark.usefixtures('X_run')
def test_ensemble(algo_metrics_init, algo_params_init, h_params_init, X_run):
    my_ensemble = Ensemble(algo_metrics_init, algo_params_init, h_params_init)

    assert len(my_ensemble.param_perms) == 4
    assert len(my_ensemble.param_perms['MiniBatchKMeans']) == 2
    assert len(my_ensemble.param_perms['SpectralClustering']) == 4
    assert len(my_ensemble.param_perms['GaussianMixture']) == 2
    assert len(my_ensemble.param_perms['linkage_vector']) == 2

    res, _ = my_ensemble(X_run, (2, 4), e_params=None, algo_selections=None)

    assert len(res) == len(algo_selections_init)  # Test that we got all the keys.
    assert all(x in res for x in algo_selections_init)  # Test exact key match.
    assert all(sub_v in res[k] for k, v in algo_metrics_init.items() for sub_v in v)  # Test that we got all metrics.
    assert all(x == 3 for vote_dict in res['MiniBatchKMeans'] for _, x in vote_dict.items())  # Check results.

@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('h_params_init')
@pytest.mark.usefixtures('X_run')
@pytest.mark.usefixtures('algo_selections_run')
def test_algo_selections(algo_metrics_init, algo_params_init, h_params_init, X_run, algo_selections_run):
    my_ensemble = Ensemble(algo_metrics_init, algo_params_init, h_params_init)

    res, _ = my_ensemble(X_run, (2, 4), e_params=None, algo_selections=algo_selections_run)

    assert len(res) == len(algo_selections_run)  # Test that we got all the keys.
    assert all(x in res for x in algo_selections_run)  # Test exact key match.
    assert all(v in res[k] for k in algo_selections_run for v in algo_metrics_init[k])  # Test that we got all metrics.
