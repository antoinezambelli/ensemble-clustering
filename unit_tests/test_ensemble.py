'''
Unit tests for ensemble.py
'''

import pytest
import warnings

from ensemble_clustering import Ensemble


@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('h_params_init')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('X_run')
def test_ensemble(algo_metrics_init, algo_params_init, h_params_init, param_perms_output, X_run):
    my_ensemble = Ensemble(algo_metrics_init, algo_params_init, h_params_init)
    assert len(my_ensemble.param_perms) == 4
    assert len(my_ensemble.param_perms['MiniBatchKMeans']) == 2
    assert len(my_ensemble.param_perms['SpectralClustering']) == 3
    assert len(my_ensemble.param_perms['GaussianMixture']) == 2
    assert len(my_ensemble.param_perms['linkage_vector']) == 2
    assert my_ensemble.param_perms == param_perms_output

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)  # Caused by Spectral due to assymmetric toy array.
        res, _ = my_ensemble(X_run, (2, 4), e_params=None, algo_selections=None)
    assert len(res) == len(algo_metrics_init)  # Test that we got all the keys.
    assert all(x in res for x, _ in algo_metrics_init.items())  # Test exact key match.
    assert all(sub_v in res[k][0] for k, v in algo_metrics_init.items() for sub_v in v)  # Test that we got all metrics.
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
    assert all(v in res[k][0] for k in algo_selections_run for v in algo_metrics_init[k])  # Test that we got all metrics.

# TODO: for all...use param_perms_output get perms. then pass in [0]'th perm and specific algo.

# TODO: main test.
# test with mini, hca, spectral.
# test with many n_c to get different numbers.
# match output keys == metrics, values == ??.

# TODO: compute_graph test. match output to explicit kneighbors(X_run)

# TODO: run_trial test...match model object to explicit Mini initialized, labels to known value.

# NOTE: can test all of them in one test if we wanted...

