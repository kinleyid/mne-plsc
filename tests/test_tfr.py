
import pytest
import mne
import mne_plsc
import numpy as np
from matplotlib import pyplot as plt

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
    freqs = np.linspace(5, 25, 20)
    montage = mne.channels.make_standard_montage('biosemi16')
    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=sfreq,
                           ch_types='eeg')
    info.set_montage(montage)
    array_data = np.random.normal(size=(2*n_ptpt, info['nchan'], len(freqs), len(times)))
    array_data[between == 'b2'] += 1
    array_data[between == 'w2'] += 1
    evoked_data = []
    for ptpt_data in array_data:
        evoked = mne.time_frequency.AverageTFRArray(data=ptpt_data,
                                                    info=info,
                                                    times=times,
                                                    freqs=freqs)
        evoked_data.append(evoked)
    return evoked_data, covariates, between, within, participant

def run_result_plots(result):
    result.plot_boot_stat(0)
    result.plot_brain_sals(lv_idx=0)
    result.plot_cluster_sizes(lv_idx=0)
    if 'plot_marginal_brain_scores' in dir(result):
        result.plot_marginal_brain_scores(lv_idx=0, margin='time')
        result.plot_marginal_brain_scores(lv_idx=0, margin='chan')
    result.plot_cluster(lv_idx=0, cluster_idx=0)
    result.plot_cluster(lv_idx=0, cluster_idx=0, highlight='extent')
    result.plot_lv(lv_idx=0)
    result.plot_scores(lv_idx=0)
    plt.close('all')

def run_result_methods(result):
    result.add_adjacency()
    result.cluster()
    result.model.permute(10)
    result.model.bootstrap(10)
    result.cluster(which='z-scores')
    run_result_plots(result)

def test_mc_both(sample_data):
    data, _, between, within, participant = sample_data
    result = mne_plsc.fit_mc(data=data,
                                 between=between,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    run_result_methods(result)

def test_mc_within(sample_data):
    data, _, between, within, participant = sample_data
    result = mne_plsc.fit_mc(data=data,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    run_result_methods(result)

def test_beh_both(sample_data):
    data, covariates, between, within, participant = sample_data
    result = mne_plsc.fit_beh(data=data,
                                  covariates=covariates,
                                  between=between,
                                  within=within,
                                  participant=participant,
                                  random_state=123)
    run_result_methods(result)
    
def test_beh_within(sample_data):
    data, covariates, between, within, participant = sample_data
    result = mne_plsc.fit_beh(data=data,
                                  covariates=covariates,
                                  within=within,
                                  participant=participant,
                                  random_state=123)
    run_result_methods(result)