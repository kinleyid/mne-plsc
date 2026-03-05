
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes

from pdb import set_trace

class MCPLS():
    def __init__(self, data, design, between=None, within=None, participant=None):
        # TODO: make sure at least one of within and between is not none
        # TODO: make sure participant is defined if within is defined
        # TODO: enfore categoricity
        # TODO: multiple within and between factors?
        # TODO: check whether there are multiple levels of within and between factors
        self.data = data
        self.design = design
        self.within = within
        self.between = between
        self.participant = participant
        # Default to a dummy participant indicator---even if there is only a between condition, still need a way to differentiate between observations
        if self.participant is None:
            self.design['participant'] = self.design.index
            self.participant = 'participant'
        # Get stratifying variable
        strat = _get_stratifier(self.design, self.between, self.within)
        self.labels = strat.unique()
        # TODO: keep track of labels
        # SVD decomposition
        u, s, v = _mc_svd(self.data, strat)
        self.design_sals = u
        self.singular_vals = s
        self.brain_sals = v.T
    def permute(self, n_perm=5000):
        perm_singvals = []
        for perm_n in tqdm(range(n_perm)):
            permuted = _get_permuted_design(
                design=self.design,
                between=self.between,
                within=self.within,
                participant=self.participant)
            strat = _get_stratifier(permuted, self.between, self.within)
            perm_singvals.append(_mc_svd(self.data, strat, compute_uv=False))
        perm_singvals = np.stack(perm_singvals)
        pvals = (np.sum(perm_singvals >= self.singular_vals, axis=0) + 1) / (n_perm + 1)
        self.pvals = pvals
        return perm_singvals
    def bootstrap(self, n_boot=5000, confint_level=0.025):
        # Get variables needed for bootstrapping
        bootstrap_vars = _setup_for_bootstrapping(self.data, self.design, self.participant, self.between)
        brain_resampled = []
        design_resampled = []
        for boot_n in tqdm(range(n_boot)):
            # Get bootstrap sample
            boot_design = _get_bootstrap_sample(*bootstrap_vars)
            # Get data matrix
            boot_data = np.stack(boot_design['data'])
            strat = _get_stratifier(boot_design, self.between, self.within)
            u, s, v = _mc_svd(boot_data, strat)
            brain_resampled.append(v.T @ np.diag(s))
            # Design saliences
            # R, _ = orthogonal_procrustes(u, self.design_sals, check_finite=False)
            # u_boot = u @ R
            # Note: need to multiply by s because, for 2 conditions, u will always be the same
            design_resampled.append(u @ np.diag(s))
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios = self.brain_sals / stds
        # Compute confidence intervals for design saliences
        self.bootstrap_ci = np.quantile(np.stack(design_resampled), [confint_level, 1 - confint_level], axis=0)
    def bootstrap_old(self, n_boot=5000, confint_level=0.025):
        # Create table containing both design info and data
        design_with_data = self.design.copy().assign(data=list(self.data))
        ptptwise_subtables = dict(tuple(design_with_data.groupby(self.participant)))
        if self.between:
            # Which participants are in which between conditions?
            groupwise_ptpts = self.design.groupby(self.between)[self.participant].apply(np.unique)
        else:
            unique_participants = self.design[self.participant].unique()
        brain_resampled = []
        design_resampled = []
        for boot_n in tqdm(range(n_boot)):
            if self.between:
                # Sample subjects within groups---keep conditions for subjects
                groupwise_resampled = groupwise_ptpts.apply(lambda x: np.random.choice(x, x.size, replace=True))
                resampled_participants = np.concatenate(groupwise_resampled.values)
            else:
                # Sample subjects
                resampled_participants = np.random.choice(unique_participants, size=unique_participants.size, replace=True)
            boot_design = pd.concat([ptptwise_subtables[ptpt] for ptpt in resampled_participants], ignore_index=True)
            boot_data = np.stack(boot_design['data'])
            strat = _get_stratifier(boot_design, self.between, self.within)
            u, s, v = _mc_svd(boot_data, strat)
            # Procrustes rotation---brain saliences
            """
            R, _ = orthogonal_procrustes(v.T, self.brain_sals, check_finite=False)
            v_boot = v.T @ R
            brain_resampled.append(v_boot)
            """
            brain_resampled.append(v.T)
            # Design saliences
            # R, _ = orthogonal_procrustes(u, self.design_sals, check_finite=False)
            # u_boot = u @ R
            design_resampled.append(u)
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios = self.brain_sals / stds
        # Compute confidence intervals for design saliences
        self.bootstrap_ci = np.quantile(np.stack(design_resampled), [confint_level, 1 - confint_level], axis=0)
   
def _get_stratifier(design, between, within):
    cols = [col for col in [between, within] if col is not None]
    if len(cols) == 0:
        # Nothing to stratify on (allowed for behavioural PLS)
        # Return dummy stratifier containing only one value
        """
        dummy = pd.DataFrame({'stratifier': [0]*len(design)})
        stratifier = pd.MultiIndex.from_frame(dummy)
        """
        stratifier = pd.Series(0, index=design.index)
    else:
        """
        stratifier = pd.MultiIndex.from_frame(design[cols])
        """
        stratifier = design[cols].apply(tuple, axis=1)
    return stratifier

def _sort_tabular():
    # TODO: tables should be sorted by group, then participant, then condition
    pass

def _get_permuted_design(design, participant, between=None, within=None):
    permuted = design.copy()
    if within is None:
        # Shuffle all rows
        permuted = permuted.sample(frac=1)
    else:
        # Organize by participant
        ptptwise_blocks = [
            block
            for _, block in permuted.groupby(participant)
        ]
        # Shuffle within participants
        ptptwise_blocks = [
            block.sample(frac=1) for block in ptptwise_blocks
        ]
        if between:
            # Shuffle entire participants
            # TODO: note that if there's a dummy between factor, this will shuffle participants even though it shouldn't
            np.random.shuffle(ptptwise_blocks)
        permuted = pd.concat(ptptwise_blocks, ignore_index=True)
       
    return permuted

def _setup_for_bootstrapping(data, design, participant, between=None):
    # Create table containing both design info and data
    design_with_data = design.copy().assign(data=list(data))
    # Get sub-tables corresponding to (possibly one, possibly multiple) observations per participant
    ptptwise_subtables = dict(tuple(design_with_data.groupby(participant)))
    if between:
        # Which participants are in which between conditions?
        groupwise_ptpts = design.groupby(between)[participant].apply(np.unique)
    else:
        # Group by a dummy variable that's the same for everyone
        # Just so we can use the same code later whether or not there's a between-participants condition
        groupwise_ptpts = design.groupby(lambda _: 0)[participant].apply(np.unique)
    return design_with_data, ptptwise_subtables, groupwise_ptpts

def _get_bootstrap_sample(dataframe, ptptwise_subtables, groupwise_ptpts):
    # Sample subjects within groups
    groupwise_resampled = groupwise_ptpts.apply(lambda x: np.random.choice(x, x.size, replace=True))
    resampled_participants = np.concatenate(groupwise_resampled.values)
    # Get data for sampled participants, including within-participant conditions 
    resample = pd.concat([ptptwise_subtables[ptpt] for ptpt in resampled_participants], ignore_index=True)
    return resample

def _mc_svd(data, strat, compute_uv=True):
    # Stack level-wise means
    level_means = np.stack([data[strat == level].mean(axis=0) for level in np.unique(strat)])
    # Mean center columns and decompose
    mean_centred = level_means - level_means.mean(axis=0)
    return np.linalg.svd(mean_centred, full_matrices=False, compute_uv=compute_uv)
        
def _beh_svd(data, cov, strat, compute_uv=True):
    # Get level-wise corelation matrices
    submatrices = []
    for level in np.unique(strat):
        idx = strat == level
        # Collect correlation matrix for this level of the stratifier
        submatrix = _corr(cov[idx], data[idx])
        submatrices.append(submatrix)
    # Stack and decompose
    R = np.concat(submatrices) # TODO: will this work with just one submatrix?
    return np.linalg.svd(R, full_matrices=False, compute_uv=compute_uv)

def _corr(X, Y):
    # Center
    Xc = X - X.mean(axis=0)
    Yc = Y - Y.mean(axis=0)
    # Covariance
    cov = Xc.T @ Yc / (X.shape[0] - 1)
    # Normalize
    stdX = X.std(axis=0, ddof=1)
    stdY = Y.std(axis=0, ddof=1)
    return cov / np.outer(stdX, stdY)

class BehPLS():
    def __init__(self, data, design, covariates=None, within=None, between=None, participant=None):
        self.data = data
        self.design = design
        self.covariates = covariates
        self.within = within
        self.between = between
        self.participant = participant # TODO: dummy participant code
        strat = _get_stratifier(self.design, self.between, self.within)
        self.labels = strat.unique()
        u, s, v = _beh_svd(data, self.design[self.covariates].to_numpy(), strat)
        self.design_sals = u
        self.singular_vals = s
        self.brain_sals = v.T
    def permute(self, n_perm=5000):
        perm_singvals = []
        print('Permuting...')
        for perm_n in tqdm(range(n_perm)):
            permuted = _get_permuted_design(
                design=self.design,
                between=self.between,
                within=self.within,
                participant=self.participant)
            strat = _get_stratifier(permuted, self.between, self.within)
            cov = permuted[self.covariates].to_numpy()
            perm_singvals.append(_beh_svd(self.data, cov, strat, compute_uv=False))
        perm_singvals = np.stack(perm_singvals)
        pvals = (np.sum(perm_singvals >= self.singular_vals, axis=0) + 1) / (n_perm + 1)
        self.pvals = pvals
        return perm_singvals
    def bootstrap(self, n_boot=5000, confint_level=0.025):
        # Get variables needed for bootstrapping
        bootstrap_vars = _setup_for_bootstrapping(self.data, self.design, self.participant, self.between)
        brain_resampled = []
        design_resampled = []
        print('Bootstrapping...')
        for boot_n in tqdm(range(n_boot)):
            # Get bootstrap sample
            boot_design = _get_bootstrap_sample(*bootstrap_vars)
            # Get data matrix
            boot_data = np.stack(boot_design['data'])
            strat = _get_stratifier(boot_design, self.between, self.within)
            u, s, v = _beh_svd(boot_data, self.design[self.covariates].to_numpy(), strat)
            brain_resampled.append(v.T)
            # Design saliences
            # R, _ = orthogonal_procrustes(u, self.design_sals, check_finite=False)
            # u_boot = u @ R
            # design_resampled.append(u)
            design_resampled.append(u @ np.diag(s))
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios = (self.brain_sals @ np.diag(self.singular_vals)) / stds
        # Compute confidence intervals for design saliences
        self.bootstrap_ci = np.quantile(np.stack(design_resampled), [confint_level, 1 - confint_level], axis=0)
"""
def _get_permutation_idx(participant, within=None, n_perm=5000):
    # Returns a list of arrays that can be use to index the original data
    
    if within is None:
        # Add dummy within-participants variable
        within = np.array([0]*len(participant))
    # A list of rows and their corresponding within-participant conditions
    rows_and_conds = list(zip(np.arange(len(participant)), within))
    set_trace()
    # Organize observations by participant
    ptptwise_rows_and_conds = []
    for curr_participant in np.unique(participants):
        
    ptptwise_rows_and_conds = [rows_and_conds[participant == idx] for idx in np.unique(participant)]
    
    set_trace()
    print('Generating indices for permuting data...')
    perm_idxs = []
    for perm_n in tqdm(range(n_perm)):
        # Shuffle participants
        ptptwise_rows_and_conds = np.random.permutation(ptptwise_rows_and_conds)
        # Shuffle within participants
        ptptwise_rows = [np.random.permutation(ptpt_rows) for ptpt_rows in ptptwise_rows]
        # Flatten and collect
        perm_idxs.append(np.hstack(ptptwise_rows))
    
    return perm_idxs
"""