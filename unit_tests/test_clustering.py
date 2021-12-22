'''
Unit tests for ensemble.py
'''

import pytest
import warnings

import numpy as np
import scipy
import sklearn

from ensemble_clustering import Clustering


@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('X_run')
def test_clustering(algo_metrics_init, algo_params_init, param_perms_output, X_run):
    my_clust = Clustering(X_run, (2, 7), algo_params_init, algo_metrics_init)

    hca_res = my_clust(param_perms_output['linkage_vector'][0], 'linkage_vector')
    assert hca_res == {'elbow': 2, 'inertia': 5, 'max_diff': 2}

    mini_res = my_clust(param_perms_output['MiniBatchKMeans'][0], 'MiniBatchKMeans')
    assert mini_res == {'aic': 5, 'bic': 5, 'inertia': 5, 'silhouette_score': 5}

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', category=UserWarning)  # Caused by Spectral due to assymmetric toy array.
        spec_res = my_clust(param_perms_output['SpectralClustering'][0], 'SpectralClustering')
    assert spec_res == {'inertia': 5}

@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('X_run')
@pytest.mark.usefixtures('graph_output')
def test_compute_graph(algo_metrics_init, algo_params_init, param_perms_output, X_run, graph_output):
    my_clust = Clustering(X_run, (2, 7), algo_params_init, algo_metrics_init)

    graph = my_clust.compute_graph(param_perms_output['SpectralClustering'][0])
    assert graph.shape == (10, 10)
    assert isinstance(graph, scipy.sparse.csr.csr_matrix)
    assert np.allclose(graph.toarray(), graph_output)

@pytest.mark.usefixtures('algo_metrics_init')
@pytest.mark.usefixtures('algo_params_init')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('X_run')
@pytest.mark.usefixtures('trial_output_labels')
def test_run_trial(algo_metrics_init, algo_params_init, param_perms_output, X_run, trial_output_labels):
    my_clust = Clustering(X_run, (2, 7), algo_params_init, algo_metrics_init)

    model, labels = my_clust.run_trial(X_run, param_perms_output['MiniBatchKMeans'][0], n_c=5, algo='MiniBatchKMeans')
    assert isinstance(model, sklearn.cluster._kmeans.MiniBatchKMeans)
    assert (np.unique(labels) == np.unique(trial_output_labels)).all()
    assert (labels[[0, 2, 3, 9]] == labels[0]).all()  # K-means labeling can change: cluster 1 -> cluster 3 on subsequent runs.
    assert (labels[[4, 6]] == labels[4]).all()
    assert (labels[[1, 7]] == labels[1]).all()
    assert sum(labels == labels[5]) == 1  # 5 and 8 are singletons.
    assert sum(labels == labels[8]) == 1
