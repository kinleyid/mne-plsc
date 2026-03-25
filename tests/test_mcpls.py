
import pytest
from mne_plsc import compute
import numpy as np

@pytest.fixture
def sample_matrix_data():
    np.random.seed(123)

    data = np.random.normal(size=(8, 2))
    covariates = np.random.normal(size=(8, 2))
    between = np.array([0]*4 + [1]*4)
    within = np.array([0, 1]*4)
    participant = np.array(np.cumsum([1, 0]*4))
    return data, covariates, between, within, participant

def test_mc(sample_matrix_data):
    # Simple testing of model fitting
    data, _, between, within, participant = sample_matrix_data
    pls = compute.MCPLS()
    pls.fit(X=data, between=between, within=within, participant=participant)
    pls.permute(n_perm=10)
    pls.bootstrap(n_boot=10)
    # Plotting
    pls.plot_lv(lv_idx=0, which='saliences')
    pls.cluster(which='saliences')
    

def test_beh(sample_matrix_data):
    data, covariates, between, within, participant = sample_matrix_data
    pls = compute.BehPLS()
    pls.fit(X=data, covariates=covariates, between=between, within=within, participant=participant)
    pls.permute(n_perm=10)
    pls.bootstrap(n_boot=10)