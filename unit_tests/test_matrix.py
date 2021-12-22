'''
Unit tests for matrix.py
'''

import pytest

from ensemble_clustering import E


@pytest.mark.usefixtures('ensemble_output')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('e_params_input')
@pytest.mark.usefixtures('E_output')
def test_E_single(ensemble_output, param_perms_output, e_params_input, E_output):
    my_E = E([ensemble_output], param_perms_output)

    E_res = my_E(e_params_input)
    assert E_res == E_output

@pytest.mark.usefixtures('ensemble_output')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('e_params_input')
@pytest.mark.usefixtures('E_output')
def test_E_multi(ensemble_output, param_perms_output, e_params_input, E_output):
    e_out_mod = ensemble_output
    e_out_mod['linkage_vector'][1] = {'elbow': 2, 'inertia': 2, 'max_diff': 2}
    my_E = E([ensemble_output, e_out_mod], param_perms_output)

    E_res = my_E(e_params_input)
    assert E_res['best_algo_global'] == E_output['best_algo_global'][0:2]
    assert len(E_res['mode']['full']['num_clusters']) == 2

@pytest.mark.usefixtures('ensemble_output')
@pytest.mark.usefixtures('param_perms_output')
@pytest.mark.usefixtures('e_params_input')
@pytest.mark.usefixtures('E_output')
def test_E_no_ground_truth(ensemble_output, param_perms_output, e_params_input, E_output):
    del e_params_input['ground_truth']
    my_E = E([ensemble_output], param_perms_output)

    E_res = my_E(e_params_input)
    assert E_res['best_algo_global'] is None
    assert E_res['best_algo'] == [None]

    E_output['best_algo_global'] = None
    E_output['best_algo'] = [None]
    assert E_res == E_output
