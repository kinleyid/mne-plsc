
import mne
import mne_plsc
import numpy as np
import pandas as pd

from pdb import set_trace

def get_epoch_labels(epochs):
    """
    Get a list of labels corresponding to each epoch for a set of epoched data.

    Parameters
    ----------
    epochs : :class:`mne.Epochs` | :class:`mne.time_frequency.EpochsSpectrum` | :class:`mne.time_frequency.EpochsTFR`
        MNE data object containing epoch data.

    Returns
    -------
    labels : ``list``
        List of the same length as the input data object containing one label per epoch.
        
    Examples
    --------
    >>> labels = get_epoch_labels(epo)
    """
    reverse_id = {v: k for k, v in epochs.event_id.items()}
    labels = [reverse_id[n] for n in epochs.events[:, 2]]
    return labels

def average_epochs_by_label(epochs_list, between=None):
    """
    From a list of epoch data, get a list of average data, one per epoch label per participant, and a design matrix.

    Parameters
    ----------
    epochs_list : :class:`mne.Epochs` | :class:`mne.time_frequency.EpochsSpectrum` | :class:`mne.time_frequency.EpochsTFR`
        MNE data object containing epoch data.
    between : iterable
        Iterable of between-participants condition labels corresponding to each element in ``epochs_list``.

    Returns
    -------
    data_list : ``list``
        List containing epoch averages
    design : `pd.DataFrame`
        Design matrix
        
    Examples
    --------
    >>> labels = get_epoch_labels(epo)
    """
    if not isinstance(epochs_list, list):
        raise ValueError('data must be a list of recordings')
    if between is not None:
        if len(epochs_list) != len(between):
            raise ValueError('epochs_list and between must be of the same length')
    data_list = []
    rows = []
    for ptpt_idx, ptpt_data in enumerate(epochs_list):
        labels = set(get_epoch_labels(ptpt_data))
        for label in labels:
            avg = ptpt_data[label].average()
            row = {'within': label, 'participant': ptpt_idx}
            if between is not None:
                row['between'] = between[ptpt_idx]
            data_list.append(avg)
            rows.append(row)
    design = pd.DataFrame.from_records(rows)
    return data_list, design

def average_epochs_by_metadata(epochs_list, column, between=None):
    """
    From a list of epoch data, get a list of average data, one per unique value in a metadata column per participant, and a design matrix.

    Parameters
    ----------
    epochs_list : :class:`mne.Epochs` | :class:`mne.time_frequency.EpochsSpectrum` | :class:`mne.time_frequency.EpochsTFR`
        MNE data object containing epoch data.
    column : ``str``
        Name of the column in each epoch's metadata containing a variable that should be used to stratify observations. I.e., the name of the column containing the within-subjects experimental condition.
    between : iterable
        Iterable of between-participants condition labels corresponding to each element in ``epochs_list``.

    Returns
    -------
    data_list : ``list``
        List containing epoch averages
    design : `pd.DataFrame`
        Design matrix
        
    Examples
    --------
    >>> labels = get_epoch_labels(epo)
    """
    if not isinstance(epochs_list, list):
        raise ValueError('data must be a list of recordings')
    if between is not None:
        if len(epochs_list) != len(between):
            raise ValueError('epochs_list and between must be of the same length')
    data_list = []
    rows = []
    for ptpt_idx, ptpt_data in enumerate(epochs_list):
        cond = ptpt_data.metadata[column]
        labels = cond.unique()
        for label in labels:
            avg = ptpt_data[cond == label].average()
            row = {'within': label, 'participant': ptpt_idx}
            if between is not None:
                row['between'] = between[ptpt_idx]
            data_list.append(avg)
            rows.append(row)
    design = pd.DataFrame.from_records(rows)
    return data_list, design

def get_datamat(data, template):
    # MNE object (or list thereof) to matrix
    if template.space == 'source' and template.domain == 'time-freq':
        # Special case: outer list is epochs or participants
        all_obs = [] # observations as umbrella term
        for curr_obs in data:
            # Get freqs x times matrix
            array = np.stack([stc.data for stc in curr_obs])
            # Flatten
            row = array.flatten()
            # Collect
            all_obs.append(row)
        datamat = np.stack(all_obs)
        if np.iscomplex(datamat).any():
            raise ValueError('Data is complex; convert to real power values before model fitting')
    else:
        if isinstance(data, list):
            # Each element is a different participant-wise average
            if template.space == 'source':
                get_data = lambda x: x.data
            else:
                get_data = lambda x: x.get_data()
            datamat = np.stack([get_data(item).flatten() for item in data])
        else:
            # Single participant data
            datamat = np.stack([epoch.flatten() for epoch in data.get_data()])
    return datamat

def get_grouping(between, within):
    # Figure out if data is grouped by a between-subjects
    # variable, a within-subjects variable, both, or neither
    if between is None and within is None:
        grouping = 'neither'
    elif between is None or within is None:
        grouping = 'within' if between is None else 'between'
    else:
        grouping = 'both'
    return grouping
    

def get_non_margin_axes(margin, datatype):
    mapping = {
        'time': {'epo': 0,
                 'tfr': (0, 1),
                 'surf-stc': 0,
                 'vol-stc': 0},
        'freq': {'spec': 0,
                 'tfr': (0, 2)},
        'chan': {'epo': 1,
                 'spec': 1,
                 'tfr': (1, 2)},
        'time-freq': {'tfr': 0}
    }
    non_margin_axes = mapping[margin][datatype]
    return non_margin_axes

def get_cluster_extent(mask):
    # Sum over spatial dimension
    mask = mask.sum(axis=0) > 0
    in_extent = np.zeros_like(mask)
    # Build a slice for each axis from its min to max (inclusive)
    true_indices = np.argwhere(mask)
    mins = true_indices.min(axis=0)
    maxs = true_indices.max(axis=0)
    lims = list(zip(mins, maxs))
    slices = tuple(slice(lo, hi + 1) for lo, hi in lims)
    in_extent[slices] = True
    return in_extent, lims