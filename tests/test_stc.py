
import pytest
import mne
import mne_plsc
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
from mne.datasets import sample
from mne.minimum_norm import read_inverse_operator

import matplotlib
matplotlib.use('Agg', force=True)

from pdb import set_trace

@pytest.fixture
def sample_data():
    np.random.seed(123)
    n_ptpt = 10
    between = ['b1']*n_ptpt + ['b2']*n_ptpt
    within = ['w1', 'w2']*n_ptpt
    participant = np.cumsum([1, 0]*n_ptpt)
    covariates = np.random.normal(size=(2*n_ptpt, 2))
    sfreq = 20
    times = np.arange(0, 1, 1/sfreq)
    n_vert = 3
    stcs = []
    for ptpt in range(n_ptpt*2):
        array_data = np.random.normal(size=(n_vert, len(times)))
        array_data = np.concat([array_data]*2)
        array_data[between == 'b2'] += 1
        array_data[between == 'w2'] += 1
        stc = mne.SourceEstimate(data=array_data,
                                 vertices=[np.arange(n_vert)]*2,
                                 tmin=times[0],
                                 tstep=1/sfreq,
                                 subject='fsaverage')
        stcs.append(stc)
    data_path = sample.data_path()
    meg_path = data_path / "MEG" / "sample"
    fname_inv = meg_path / "sample_audvis-meg-oct-6-meg-inv.fif"
    inverse_operator = read_inverse_operator(fname_inv)
    src = inverse_operator['src']
    return stcs, covariates, between, within, participant, src

def run_result_plots(result):
    result.plot_boot_stat(0)
    result.plot_brain_sals(lv_idx=0)
    # result.plot_cluster_sizes(lv_idx=0)
    if 'plot_marginal_brain_scores' in dir(result):
        result.plot_marginal_brain_scores(lv_idx=0, margin='time')
    # Can't test cluster visualization
    result.plot_lv(lv_idx=0)
    result.plot_scores(lv_idx=0)
    plt.close('all')

def run_result_methods(result, src):
    result.add_source_info(src=src)
    result.add_adjacency()
    # result.cluster()
    result.model.permute(10)
    result.model.bootstrap(10)
    # result.cluster(which='z-scores')
    run_result_plots(result)

def test_mc_both(sample_data):
    data, _, between, within, participant, src = sample_data
    result = mne_plsc.fit_mc(data=data,
                                 between=between,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    run_result_methods(result, src)

def test_mc_within(sample_data):
    data, _, between, within, participant, src = sample_data
    result = mne_plsc.fit_mc(data=data,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    run_result_methods(result, src)

def test_beh_args(sample_data):
    # Test different methods of providing data
    data, covariates, between, within, participant, src = sample_data
    mne_plsc.fit_beh(data=data,
                     covariates=covariates[:, 0],
                     random_state=123)
    design = pd.DataFrame(covariates,
                          columns=['cov1', 'cov2'])
    design['group'] = between
    design['cond'] = within
    design['ptpt'] = participant
    mne_plsc.fit_beh(data=data,
                     design=design,
                     covariates='cov1',
                     between='group',
                     within='cond',
                     participant='ptpt',
                     random_state=123)
    mne_plsc.fit_beh(data=data,
                     covariates=design['cov1'],
                     between=design['group'],
                     within=design['cond'],
                     participant=design['ptpt'],
                     random_state=123)

def test_mc_args(sample_data):
    # Test different methods of providing data
    data, covariates, between, within, participant, src = sample_data
    design = pd.DataFrame(covariates,
                          columns=['cov1', 'cov2'])
    design['group'] = between
    design['cond'] = within
    design['ptpt'] = participant
    mne_plsc.fit_mc(data=data,
                    design=design,
                    between='group',
                    within='cond',
                    participant='ptpt',
                    random_state=123)
    mne_plsc.fit_mc(data=data,
                    between=design['group'],
                    within=design['cond'],
                    participant=design['ptpt'],
                    random_state=123)

def test_beh_both(sample_data):
    data, covariates, between, within, participant, src = sample_data
    result = mne_plsc.fit_beh(data=data,
                              covariates=covariates,
                              between=between,
                              within=within,
                              participant=participant,
                              random_state=123)
    run_result_methods(result, src)
    
def test_beh_within(sample_data):
    data, covariates, between, within, participant, src = sample_data
    result = mne_plsc.fit_beh(data=data,
                              covariates=covariates,
                              within=within,
                              participant=participant,
                              random_state=123)
    run_result_methods(result, src)