
import pytest
import mne
import mne_plsc
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd

import matplotlib
matplotlib.use('Agg', force=True)

from pdb import set_trace

@pytest.fixture
def sample_data():
    np.random.seed(123)
    n_ptpts = 10
    trials_per_cond = 10
    sfreq = 20
    times = np.arange(0, 1, 1/sfreq)
    montage = mne.channels.make_standard_montage('biosemi16')
    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=sfreq,
                           ch_types='eeg')
    info.set_montage(montage)
    all_epochs = []
    for ptpt in range(n_ptpts):
        array_data = np.random.normal(size=(2*n_ptpts, info['nchan'], len(times)))
        metadata = pd.DataFrame({
            'cond': [0, 1]*trials_per_cond,
            'cov_1': np.random.normal(size=(2*trials_per_cond)),
            'cov_2': np.random.normal(size=(2*trials_per_cond))})
        epochs = mne.EpochsArray(data=array_data,
                                 info=info,
                                 metadata=metadata)
        all_epochs.append(epochs)
    return all_epochs

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

def test_utils(sample_data):
    mne_plsc.utils.average_epochs_by_label(sample_data)
    mne_plsc.utils.average_epochs_by_metadata(sample_data, column='cond')

def test_within(sample_data):
    result = mne_plsc.fit_within_beh(data=sample_data,
                                     within='cond',
                                     covariates=['cov_1', 'cov_2'],
                                     random_state=123)
    run_result_methods(result)
