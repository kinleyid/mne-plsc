
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.linalg import orthogonal_procrustes

from pdb import set_trace

class MCPLS():
    def __init__(self, subtract=None):
        self.subtract = subtract
    def fit(self, X, between=None, within=None, participant=None):
        if participant is None and within is not None:
            raise ValueError('Participants must be differentiated if there is a within-participants factor')
        self.X_ = X
        self.set_up_indicators(between=between, within=within, participant=participant)
        # TODO: make sure at least one of within and between is not none
        # TODO: make sure participant is defined if within is defined
        # TODO: enfore categoricity
        # TODO: multiple within and between factors?
        # TODO: check whether there are multiple levels of within and between factors
        # TODO: check whether subtract option is possible given availability of factors
        # TODO: make sure lengths of inputs are all the same
        # TODO: enforce one between condition per participant
        # Get stratifying variable
        # TODO: keep track of labels
        # SVD decomposition
        mean_centred = _get_mean_centred(
            X=self.X_,
            between=self.between_,
            within=self.within_,
            participant=self.participant_,
            subtract=self.subtract)
        u, s, v = np.linalg.svd(mean_centred, full_matrices=False, compute_uv=True)
        self.design_sals_ = u
        self.contrast_ = u @ np.diag(s)
        self.singular_vals_ = s
        self.variance_explained_ = s / sum(s)
        self.brain_sals_ = v.T
        return self
    def permute(self, n_perm=5000):
        if n_perm < 1:
            raise ValueError('n_perm must be a positive integer')
        perm_singvals = []
        # TODO: I think you could create the between-within stratifiers here instead of every time we call _get_mean_centred for speed
        # Get indicators to permute
        to_permute = _get_matrix_to_permute(
            between=self.between_,
            within=self.within_,
            participant=self.participant_)
        print('Permuting...')
        for perm_n in tqdm(range(n_perm)):
            # Permute
            perm_idx = _get_permutation(len(self.X_), between=self.between_, participant=self.participant_)
            if self.within_ is not None:
                
            # Unpack permuted indicators
            if self.within_ is not None:
                within, participant = permuted[:, :2].T
                if self.between_ is not None:
                    between = permuted[:, 2]
                else:
                    between = None
            else:
                between = permuted
                within, participant = None, None
            # Run decomposition
            mean_centred = _get_mean_centred(
                X=self.X_,
                between=between,
                within=within,
                participant=participant,
                subtract=self.subtract)
            singval = np.linalg.svd(
                mean_centred,
                full_matrices=False,
                compute_uv=False)
            perm_singvals.append(singval)
        perm_singvals = np.stack(perm_singvals)
        pvals = (np.sum(perm_singvals >= self.singular_vals_, axis=0) + 1) / (n_perm + 1)
        pvals[-1] = np.nan # Last p value is not applicable
        self.pvals_ = pvals
        return perm_singvals
    def bootstrap(self, n_boot=5000, confint_level=0.025):
        self.n_boot_ = n_boot
        self.confint_level_ = confint_level
        # Get variables needed for bootstrapping
        # bootstrap_vars = _setup_for_bootstrapping(self.X_, self.design, self.participant, self.between)
        resample_vars = _get_vars_for_resampling(
            n_rows=len(self.X_),
            between=self.between_,
            within=self.within_,
            participant=self.participant_)
        brain_resampled = []
        design_resampled = []
        # Pre-allocate data
        resampled_X = np.empty_like(self.X_)
        if self.between_ is None:
            resampled_between = None
        else:
            resampled_between = np.empty_like(self.between_)
        if self.within_ is None:
            resampled_within, resampled_participant = None, None
        else:
            resampled_within = np.empty_like(self.within_)
            resampled_participant = np.empty_like(self.participant_)
        print('Bootstrap resampling...')
        for boot_n in tqdm(range(n_boot)):
            # Get indices of resample
            resample_idx = _get_resample_idx(*resample_vars)
            resample_idx = np.arange(len(self.X_))
            set_trace()
            # Get resampled data and indicators
            resampled_X[:] = self.X_[resample_idx]
            if self.between_ is not None:
                resampled_between[:] = self.between_[resample_idx]
            if self.within_ is not None:
                resampled_within[:] = self.within_[resample_idx]
                resampled_participant[:] = self.participant_[resample_idx]
            # Run decomposition
            mean_centred = _get_mean_centred(
                X=resampled_X,
                between=resampled_between,
                within=resampled_within,
                participant=resampled_participant,
                subtract=self.subtract)
            u, s, v = np.linalg.svd(
                mean_centred,
                full_matrices=False,
                compute_uv=True)
            set_trace()
            v = v.T
            # Rotate to align with original decomposition
            R, _ = orthogonal_procrustes(v, self.brain_sals_, check_finite=False)
            v = v @ R
            # Collect
            brain_resampled.append(v @ np.diag(s))
            # Brain scores
            scores = mean_centred @ self.brain_sals_
            design_resampled.append(scores)
        # Compute standard deviations for brain saliences to get bootstrap ratios
        stds = np.stack(brain_resampled).std(axis=0)
        self.bootstrap_ratios_ = (self.brain_sals_ @ np.diag(self.singular_vals_)) / stds
        # Compute confidence intervals for design saliences
        self.bootstrap_ci_ = np.quantile(np.stack(design_resampled), [confint_level, 1 - confint_level], axis=0)
    def set_up_indicators(self, between=None, within=None, participant=None):
        # TODO: ensure that if group id is higher, ptpt id is higher
        # Assign none if absent, otherwise assign integer labels
        if between is None:
            self.between_ = None
        else:
            _, self.between_ = np.unique(between, return_inverse=True)
        if within is None:
            self.within_ = None
            self.participant_ = None
        else:
            _, self.within_ = np.unique(within, return_inverse=True)
            _, self.participant_ = np.unique(participant, return_inverse=True)
        # Sort by between, then within, then participant
        if self.within_ is not None:
            sort_key = (self.within_, self.participant_)
            if self.between_ is not None:
                sort_key += (self.between_,)
            sort_idx = np.lexsort(sort_key)
            self.within_ = self.within_[sort_idx]
            self.participant_ = self.participant_[sort_idx]
        else:
            sort_idx = np.argsort(self.between_)
        if self.between_ is not None:
            self.between_ = self.between_[sort_idx]
        self.X_ = self.X_[sort_idx]
  
class BehPLS():
    def __init__(self):
        # No initialization variables
        pass
    def fit(self, X, covariates, within=None, between=None, participant=None):
        # Store data
        self.X_ = X
        self.covariates_ = covariates
        _set_up_indicators(self, between=between, within=within, participant=participant)
        stratifier = _get_stratifier(len(self.X_), self.between_, self.within_, self.participant_)
        R = _get_stacked_cormats(self.X_, self.covariates_, stratifier)
        u, s, v = np.linalg.svd(R, full_matrices=False, compute_uv=True)
        self.design_sals_ = u
        self.singular_vals_ = s
        self.brain_sals_ = v.T
    def permute(self, n_perm=5000):
        perm_singvals = []
        print('Permuting...')
        to_permute = _get_matrix_to_permute(
            between=self.between_,
            within=self.within_,
            participant=self.participant_,
            covariates=self.covariates_)
        for perm_n in tqdm(range(n_perm)):
            # Permute
            permuted = _get_permutation(to_permute, between=self.between_, participant=self.participant_)
            # Unpack permuted indicators
            if self.within_ is not None:
                within, participant = permuted[:, :2].T
                if self.between_ is not None:
                    between = permuted[:, 2]
                    covariates = permuted[:, 3:]
                else:
                    between = None
                    covariates = permuted[:, 2:]
            else:
                between = permuted[:, 0]
                within, participant = None, None
                covariates = permuted[:, 1:]
            
            
            
            permuted = _get_permutation(
                to_permute,
                between=self.between_,
                participant=self.participant_)
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

def _get_permutation(n_obs, between=None, participant=None):
    if participant is None:
        # No between-participant conditions---just shuffle all rows
        perm_idx = np.random.permutation(n_obs)
    else:
        if between is not None:
            # Shuffle participants
            n_participants = participant.max() + 1
            participant_permutation = np.random.permutation(n_participants)
            # This next line works because "participant" is both an array of
            # integer labels and an integer index that could be used to index
            # an array of unique participant IDs
            participant = participant_permutation[participant]
        # Shuffle within participants
        perm_idx = np.lexsort((np.random.rand(len(participant)), participant))
    return perm_idx

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

def _get_stratifier(n_obs, between=None, within=None, participant=None):
    if between is not None and within is not None:
        stratifier = np.column_stack((between, within))
        _, stratifier = np.unique(stratifier, axis=0, return_inverse=True) # Would be nice to do this earlier 
    elif between is not None:
        stratifier = between
    elif within is not None:
        stratifier = within
    else:
        stratifier = np.zeros((n_obs,), dtype=np.int64)
    return stratifier

def _get_mean_centred(X, between, within, participant, subtract):
    if subtract is not None:
        # Pre-subtract between- or within-wise means if applicable
        if subtract == 'between':
            group_idx = between
        elif subtract == 'within':
            group_idx = within
        rowwise_group_means = _get_groupwise_means(X, group_idx)[group_idx]
        X = X - rowwise_group_means
    # Compute group-wise means
    stratifier = _get_stratifier(len(X), between, within, participant)
    groupwise_means = _get_groupwise_means(X, stratifier)
    # Mean centre
    mean_centred = groupwise_means - groupwise_means.mean(axis=0)
    return mean_centred

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

def _get_groupwise_means(X, group_idx):
    n_groups = group_idx.max() + 1
    # Sums per group
    groupwise_sums = np.zeros((n_groups, X.shape[1]))
    np.add.at(groupwise_sums, group_idx, X)
    # Counte per group
    groupwise_counts = np.zeros(n_groups)
    np.add.at(groupwise_counts, group_idx, 1)
    # Means per group
    groupwise_means = groupwise_sums / groupwise_counts[:, None]    
    return groupwise_means

def _build_model_matrix(covariates=None, between=None, within=None, participant=None):
    # Build matrix containing indicators and covariates, if any
    # Order is ([within, participant,] [between,] [covariates])
    if within is not None:
        # If there is a within-participants condition, we need to keep
        # track of it as well as participant identity
        columns = (within, participant)
        if between is not None:
            columns += (between,)
    else:
        # Otherwise we only need to keep track of between-participants
        # condition
        columns = (between,)
    if covariates is not None:
        columns += covariates
    matrix = np.column_stack(columns)
    return matrix

def _get_vars_for_resampling(n_rows, between=None, within=None, participant=None):
    # Set up variables used for resampling
    row_idx = np.arange(n_rows)
    # Set up dummy indicators if needed
    if within is None:
        within = np.array([0]*n_rows, dtype=np.int64)
        participant = np.arange(n_rows)
    if between is None:
        between = np.array([0]*n_rows, dtype=np.int64)
    row_idx_by_participant = np.split(row_idx, np.cumsum(np.bincount(participant)[:-1]))
    between_by_participant = between[np.cumsum(np.bincount(participant)) - 1]
    participants_by_between = np.split(
        np.arange(len(row_idx_by_participant)),
        np.cumsum(np.bincount(between_by_participant)[:-1])
    )
    participant_offsets = np.cumsum([0] + [len(r) for r in row_idx_by_participant])
    return row_idx, participants_by_between, participant_offsets

def _get_resample_idx(row_idx, participants_by_between, participant_offsets):
    sampled_rows = []
    for ps in participants_by_between:
        samp = ps[np.random.randint(len(ps), size=len(ps))]
        # sampled_rows.extend(row_idx_by_participant[p] for p in samp)
        sampled_rows.extend(row_idx[participant_offsets[p]:participant_offsets[p+1]] for p in samp)
    resample_idx = np.concatenate(sampled_rows)
    return resample_idx

def _set_up_indicators(obj, n_obs, between=None, within=None, participant=None):
    # TODO: ensure that if group id is higher, ptpt id is higher
    
    # Assign none if absent, otherwise assign integer labels
    if between is None:
        obj.between_ = None
    else:
        _, obj.between_ = np.unique(between, return_inverse=True)
    if within is None:
        obj.within_ = None
        obj.participant_ = None
    else:
        _, obj.within_ = np.unique(within, return_inverse=True)
        _, obj.participant_ = np.unique(participant, return_inverse=True)
    
    # Sort by between, then within, then participant, if applicable
    if obj.between_ is None and obj.within_ is None:
        return np.arange(n_obs)
    else:
        if obj.within_ is not None:
            sort_key = (obj.within_, obj.participant_)
            if obj.between_ is not None:
                sort_key += (obj.between_,)
            sort_idx = np.lexsort(sort_key)
            obj.within_ = obj.within_[sort_idx]
            obj.participant_ = obj.participant_[sort_idx]
        else:
            sort_idx = np.argsort(obj.between_)
        if obj.between_ is not None:
            obj.between_ = obj.between_[sort_idx]
        # Requires that object already has X_
        obj.X_ = obj.X_[sort_idx]

def _get_matrix_to_permute(between=None, within=None, participant=None, covariates=None):
    cols_to_permute = ()
    if within is not None:
        # If there is a within-participants condition, we need to keep
        # track of it as well as participant identity
        cols_to_permute += (within, participant)
    if between is not None:
        cols_to_permute += (between,)
    if covariates is not None:
        cols_to_permute += covariates
    return np.column_stack(cols_to_permute)

def _get_stacked_cormats(X, covariates, stratifier):
    submatrices = []
    for level in np.unique(stratifier):
        idx = stratifier == level
        submatrix = _corr(X[idx], covariates[idx])
        submatrices.append(submatrix)
    R = np.concat(submatrices)
    return R
