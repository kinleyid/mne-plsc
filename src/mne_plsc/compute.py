
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
        self.data = data
        self.design = design
        self.within = within
        self.between = between
        self.participant = participant
        # Default to a dummy participant indicator---even if there is only a between condition, still need a way to differentiate between observations
        if self.participant is None:
            self.design['participant'] = self.design.index
            self.participant = ['participant']            
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
            permuted = _get_permuted_design(self.design, self.between, self.within, self.participant)
            strat = _get_stratifier(permuted, self.between, self.within)
            perm_singvals.append(_mc_svd(self.data, strat, compute_uv=False))
        perm_singvals = np.stack(perm_singvals)
        pvals = (np.sum(perm_singvals >= self.singular_vals, axis=0) + 1) / (n_perm + 1)
        self.pvals = pvals
    def bootstrap(self, n_boot=5000, confint_level=0.025):
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
    return pd.MultiIndex.from_frame(design[cols])
   
def _get_permuted_design(design, between, within, participant):
    permuted = design.copy()
    if between:
        # Shuffle between-participants factor
        participant_groups = (
            design[[participant, between]]
            .drop_duplicates()
            .set_index(participant)
        )
        participant_groups[between] = np.random.permutation(participant_groups[between].values)
        permuted = permuted.drop(columns=[between]).merge(
            participant_groups,
            left_on=participant,
            right_index=True
        )
    if within:
        permuted[within] = permuted.groupby(participant)[within].transform(np.random.permutation)
    return permuted

def _mc_svd(data, strat, compute_uv=True):
    # Get level-wise means
    level_means = np.stack([data[strat == level].mean(axis=0) for level in np.unique(strat)])
    # Mean center
    mean_centred = level_means - level_means.mean(axis=0)
    return np.linalg.svd(mean_centred, full_matrices=False, compute_uv=compute_uv)
        
class BehPLS():
    def __init__(self, data, cov, within, between):
        pass