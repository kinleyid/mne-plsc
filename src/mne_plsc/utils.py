
import mne
import numpy as np
import pandas as pd
import pyplsc

from pdb import set_trace

def get_epoch_labels(data):
    reverse_id = {v: k for k, v in data.event_id.items()}
    labels = [reverse_id[n] for n in data.events[:, 2]]
    return labels

def get_template(data, datatype):
    # Given a list of observations, get a single "template"
    # with info for plotting etc. later
    
    if isinstance(data, list):
        # Should already be average over epochs
        template = data[0]
    else:
        # Single-participant data; need to compute average over epochs
        if datatype == 'epo':
            # Epochs to Evoked
            template = mne.EvokedArray(
                data=np.zeros(data._data.shape[1:]),
                info=data.info,
                tmin=data.tmin)
        elif datatype in ['tfr', 'spec']:
            # EpochsTFR to AverageTFR
            template = data.average()
        else:
            pass
            # TODO: validate datatype
    
    return template

def infer_datatype(data):
    attrs = dir(data)
    if 'times' not in attrs:
        datatype = 'spec'
    else:
        if 'freqs' in attrs:
            datatype = 'tfr'
        else:
            datatype = 'epo'
    return datatype

def is_epochs(data, datatype=None):
    if datatype is None:
        datatype = infer_datatype(data)
    if datatype in ['epo', 'spec']:
        if data._data.ndim == 3:
            val = True
        else:
            val = False
    elif datatype == 'tfr':
        if data._data.ndim == 4:
            val = True
        else:
            val = False
    return val

def get_datamat(data):
    # MNE object (or list thereof) to matrix
    # TODO: validation
    if isinstance(data, list):
        # Each element is a different participant-wise average
        datamat = np.stack([item._data.flatten() for item in data])
    else:
        # Single participant data
        datamat = np.stack([epoch.flatten() for epoch in data._data])
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
                 'tfr': (0, 1)},
        'freq': {'spec': 0,
                 'tfr': (0, 2)},
        'chan': {'epo': 1,
                 'spec': 1,
                 'tfr': (1, 2)},
        'time-freq': {'tfr': 0}
    }
    non_margin_axes = mapping[margin][datatype]
    return non_margin_axes