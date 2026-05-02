
import mne
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

def infer_datatype(data):
    attrs = dir(data)
    if 'times' not in attrs:
        datatype = 'spec'
    else:
        if 'vertices' in attrs:
            if 'as_volume' in attrs:
                datatype = 'vol-stc'
            else:
                datatype = 'surf-stc'
        elif 'freqs' in attrs:
            datatype = 'tfr'
        else:
            datatype = 'epo'
    return datatype

def is_epochs(data, datatype=None):
    '''
    if datatype is None:
        datatype = infer_datatype(data)
    if datatype in ['epo', 'spec']:
        if data._data.ndim == 3:
            out = True
        else:
            out = False
    elif datatype == 'tfr':
        if data._data.ndim == 4:
            out = True
        else:
            out = False
    '''
    out = '__len__' in dir(data) # Only epoch objects have a length
    return out

def get_datamat(data, datatype):
    # MNE object (or list thereof) to matrix
    if isinstance(data, list):
        # Each element is a different participant-wise average
        if datatype in ['surf-stc', 'vol-stc']:
            get_data = lambda x: x.data
        else:
            get_data = lambda x: x.get_data()
        datamat = np.stack([get_data(item).flatten() for item in data])
    else:
        # Single participant data
        datamat = np.stack([epoch.flatten() for epoch in data.get_data()])
    return datamat

def get_indicators(design=None, between=None, within=None, participant=None):
    indicators = dict(
        between=between,
        within=within,
        participant=participant)
    if design is not None:
        for colname in indicators:
            if indicators[colname] is not None:
                indicators[colname] = design[colname]
    # Convert to pd.Categorical and get integer labels and category labels
    labels = dict()
    for k in indicators:
        if indicators[k] is not None:
            # Convert to categories
            indicators[k] = pd.Categorical(indicators[k])
            if k != 'participant': # Don't need to keep track of participants
                labels[k] = indicators[k].categories
            indicators[k] = indicators[k].codes
    indicators = tuple(indicators[k] for k in ['between', 'within', 'participant'])
    return labels, indicators

def get_grouping_old(labels):
    # Figure out if data is grouped by a between-subjects
    # variable, a within-subjects variable, both, or neither
    if 'between' in labels and 'within' in labels:
        grouping = 'both'
    elif 'between' in labels or 'within' in labels:
        grouping = 'between' if 'between' in labels else 'within'
    else:
        grouping = 'neither'
    return grouping

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
                 'surf-src': 0,
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

def get_1d_lims(bool_array):
    diff = np.diff(bool_array.astype(int))
    start = np.where(diff == 1)[0] + 1
    end = np.where(diff == -1)[0]
    if bool_array[0]:
        start = np.r_[0, start]
    if bool_array[-1]:
        end = np.r_[end, len(bool_array) - 1]
    return start, end

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