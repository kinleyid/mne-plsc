
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
    n_ptpt = 10
    between = ['b1']*n_ptpt + ['b2']*n_ptpt
    within = ['w1', 'w2']*n_ptpt
    participant = np.cumsum([1, 0]*n_ptpt)
    covariates = np.random.normal(size=(2*n_ptpt, 2))
    sfreq = 20
    times = np.arange(0, 1, 1/sfreq)
    montage = mne.channels.make_standard_montage('biosemi16')
    info = mne.create_info(ch_names=montage.ch_names,
                           sfreq=sfreq,
                           ch_types='eeg')
    info.set_montage(montage)
    array_data = np.random.normal(size=(2*n_ptpt, info['nchan'], len(times)))
    array_data[between == 'b2'] += 1
    array_data[between == 'w2'] += 1
    evoked_data = []
    for ptpt_data in array_data:
        evoked = mne.EvokedArray(data=ptpt_data, info=info)
        evoked_data.append(evoked)
    return evoked_data, covariates, between, within, participant

def test_errs(sample_data):
    data, _, between, within, participant = sample_data
    res = mne_plsc.fit_mc(data=data,
                                 between=between,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    with pytest.raises(Exception):
        res.add_source_info()
    with pytest.raises(Exception):
        res.cluster()
    res.add_adjacency()
    with pytest.raises(Exception):
        res.cluster(which='z-scores')
    with pytest.raises(Exception):
        res.get_cluster_sizes(0)
    with pytest.raises(Exception):
        res.get_cluster_data(0)
