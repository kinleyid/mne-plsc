
import pytest
import mne
import mne_plsc
import numpy as np
import os
from pathlib import Path

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

@pytest.fixture
def fit_mod(sample_data):
    data, _, between, within, participant = sample_data
    result = mne_plsc.fit_mc(data=data,
                                 between=between,
                                 within=within,
                                 participant=participant,
                                 random_state=123)
    return result

def test_io_str(fit_mod):
    fit_mod.save('tmp')
    mne_plsc.load('tmp')
    os.remove('tmp.xz')

def test_io_path(fit_mod):
    path = Path('tmp')
    fit_mod.save(path)
    mne_plsc.load(path)
    os.remove('tmp.xz')