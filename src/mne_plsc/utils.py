
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

def average_epochs_by_metadata(epochs_list, columns, between=None):
    """
    From a list of epoch data, get a list of average data, one per unique value in a metadata column per participant, and a design matrix.

    Parameters
    ----------
    epochs_list : :class:`mne.Epochs` | :class:`mne.time_frequency.EpochsSpectrum` | :class:`mne.time_frequency.EpochsTFR`
        MNE data object containing epoch data.
    column : ``str`` | iterable of column names
        Name of the column(s) in each epoch's metadata containing a variable or set of variables that should be used to stratify observations. E.g., the name of the column containing the within-subjects experimental condition.
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
        set_trace()
        # multiindex indices
        if isinstance(columns, str):
            cond = ptpt_data.metadata[columns]
        else:
            cond = ptpt_data.metadata[columns].astype(str).agg('_'.join, axis=1)
        # cond = ptpt_data.metadata[column]
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

def covariates_from_metadata(data_list, col_names):
    # TODO: document
    # Does this function really need to exist?
    cov_subtables = []
    for entry in data_list:
        subtable = entry.metadata[col_names]
        cov_subtables.append(subtable)
    covs = pd.concat(cov_subtables)
    return covs

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
        single_or_multi = 'ambig' # Not clear if list elements are participants or epochs
    else:
        if isinstance(data, list):
            # Each element is from a different participant
            if template.space == 'source':
                # get_data() not defined for source space
                get_data = lambda x: x.data
                single_or_multi = 'ambig' # Not clear if list elements are participants or epochs
            else:
                get_data = lambda x: x.get_data()
                single_or_multi = 'multi'
            datamat = np.stack([get_data(item).flatten() for item in data])
        else:
            # Single participant data
            single_or_multi = 'single'
            # Cannot be source space, therefore it's ok to use get_data() method
            datamat = np.stack([epoch.flatten() for epoch in data.get_data()])
    return datamat, single_or_multi

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

def determine_vertex_hemisphere(vert, template):
    if vert < template.vertices[0].size:
        hemi = 'lh'
    else:
        hemi = 'rh'
    return hemi

from typing import Literal

OBS_LEVEL = Literal['participant', 'condition', 'cond', 'within', 'trial']

def standardize_input(data, obs_level, between, within, participant, template, metadata_list, covariates=None):
    """
    Convert MNE data objects to a (n_obs, n_features) matrix and a label DataFrame.

    Parameters
    ----------
    data : MNE object or list of MNE objects
        For obs_level='participant': list of per-participant averages.
        For obs_level='condition':  list or other structure where each observation
                                    is a participant × condition average; participant
                                    IDs must be supplied explicitly via 'participant'.
        For obs_level='trial':      list of per-participant Epochs objects; participant
                                    identity is inferred from list position.
    obs_level : {'participant', ''within'/condition'/'cond', 'trial'}
        Observation level, i.e., granularity of each row in the output matrix.
    between : array-like or None
        Participant-level group labels, length == n_participants.
        At obs_level='participant', length == n_obs.
    within : array-like or None
        Within-participants conditions.
        Required for obs_level 'condition'.
    participant : array-like or None
        Participant ID per observation, length == n_obs.
        Required for obs_level='condition'.
        Must not be provided for obs_level='trial' (inferred from data list).
        Optional for obs_level='participant' (defaults to 0, 1, 2, ...).
    template : object
        Carries .space ('source'|'sensor') and .domain ('time'|'time-freq'),
        plus frequency/time metadata.

    Returns
    -------
    datamat : np.ndarray, shape (n_obs, n_features)
    labels  : pd.DataFrame
        Columns: 'participant' always present.
                 'condition' added for obs_level 'condition' and 'trial'.
                 'trial' added for obs_level 'trial'.
                 'group' added if 'between' is provided at sub-participant levels.
    """
    
    # Get observation level
    if obs_level in ('cond', 'within'):
        obs_level = 'condition'
    if obs_level not in ('participant', 'condition', 'trial'):
        raise ValueError(
            f"obs_level must be 'participant', 'condition'/'cond'/'within', or "
            f"'trial'; got {obs_level!r}."
        )

    # MNE object(s) to data matrix, (n. obs × n. features)
    if template.space == 'source':
        get_data = lambda x: x.data
    else:
        get_data = lambda x: x.get_data()
    if template.space == 'source' and template.domain == 'time-freq':
        # List of lists; outer list is observations, inner list is frequencies
        all_obs = []
        for curr_obs in data:
            array = np.stack([get_data(stc) for stc in curr_obs])
            all_obs.append(array.flatten())
        datamat = np.stack(all_obs)
    elif isinstance(data, list):
        # Each list element is an MNE object
        if obs_level == 'trial':
            # Don't flatten over epoch level
            submatrices = []
            for item in data:
                submatrix = get_data(item)
                submatrix = np.stack([trial.flatten() for trial in submatrix])
                submatrices.append(submatrix)
            datamat = np.concat(submatrices)
        else:
            # Fine to flatten over epoch level
            datamat = np.stack([get_data(item).flatten() for item in data])
    else:
        # Single-subject analysis; obs level has to be trial
        if obs_level != 'trial':
            raise ValueError(
                "If only one object is provided (not a list), obs_level must be 'trial'"
            )
        datamat = np.stack([epoch.flatten() for epoch in get_data(data)])

    if np.iscomplexobj(datamat):
        raise ValueError(
            "Data is complex; convert to real power values before model fitting."
        )
    n_obs = len(datamat)

    # Data labels to array
    # between     = np.asarray(between)     if between     is not None else None
    # within      = np.asarray(within)      if within      is not None else None
    # participant = np.asarray(participant) if participant is not None else None

    # Build data labels dataframe
    label_dict: dict = {}
    if obs_level == 'participant':
        if within is not None:
            raise ValueError(
                "'within' was provided. There can be no within-participants conditions when obs_level='participant'"
            )
        if between is not None and len(between) != n_obs:
            raise ValueError(
                f"obs_level='participant': 'between' must have one entry per "
                f"observation (n_obs={n_obs}), got {len(between)}."
            )
        if participant is not None and len(participant) != n_obs:
            raise ValueError(
                f"obs_level='participant': 'participant' must have length "
                f"n_obs={n_obs}, got {len(participant)}."
            )
        if covariates is not None and len(covariates) != n_obs:
            raise ValueError(
                f"obs_level='participant': 'covariates' must have length "
                f"n_obs={n_obs}, got {len(covariates)}."
            )
        if between is not None:
            label_dict['between'] = between
        if participant is None:
            participant = np.arange(n_obs)
        label_dict['participant'] = participant
        if covariates is not None:
            covariate_table = covariates
            
    elif obs_level == 'condition':
        if participant is None:
            raise ValueError(
                "obs_level='condition' requires 'participant': a per-observation "
                "array of participant IDs (length == n_obs)."
            )
        if len(participant) != n_obs:
            raise ValueError(
                f"obs_level='condition': 'participant' must have length "
                f"n_obs={n_obs}, got {len(participant)}."
            )
        if within is None:
            raise ValueError(
                "obs_level='condition' requires within-participant condition labels, specified via 'within' argument."
            )
        if len(within) != n_obs:
            raise ValueError(
                f"obs_level='condition': 'within' must have length "
                f"n_obs={n_obs}, got {len(within)}."
            )
        if between is not None and len(between) != len(np.unique(participant)):
            raise ValueError(
                f"obs_level='condition': 'between' must have one entry per unique "
                f"participant ({len(np.unique(participant))}), got {len(between)}."
            )
        if covariates is not None and len(covariates) != n_obs:
            raise ValueError(
                f"'covariates' must have length "
                f"n_obs={n_obs}, got {len(covariates)}."
            )
        
        if between is not None:
            between_map = dict(zip(np.unique(participant), between))
            label_dict['between'] = np.array([between_map[p] for p in participant])
        label_dict['participant'] = participant
        label_dict['within'] = within

    elif obs_level == 'trial':
        if participant is not None:
            raise ValueError(
                "obs_level='trial': 'participant' is inferred from the data list "
                "and should not be provided explicitly."
            )
        if within is not None and not isinstance(within, str):
            raise ValueError(
                "obs_level='trial': 'within' must be specified as a string"
            )
        if between is not None and len(between) != len(data):
            raise ValueError(
                f"obs_level='trial': 'between' must have one entry per participant "
                f"(len(data)={len(data)}), got {len(between)}."
            )
        if covariates is not None:
            cov_error = False
            if isinstance(covariates, list):
                if not all(isinstance(c, str) for c in covariates):
                    cov_error = True
            elif not isinstance(covariates, str):
                cov_error = True
            if cov_error:
                raise ValueError(
                    "obs_level='trial': 'covariates' must be specified as a string"
                )
                
        # Infer participant IDs from list structure
        if isinstance(data, list):
            n_ptpt = len(data)
            trials_per_participant = [get_data(item).shape[0] for item in data]
        else:
            n_ptpt = 1
            trials_per_participant = [get_data(data).shape[0]]
        ptpt_ids = np.repeat(np.arange(n_ptpt), trials_per_participant)
        if between is not None:
            between_map = dict(zip(np.arange(len(data)), between))
            label_dict['between'] = np.array([between_map[p] for p in ptpt_ids])
        label_dict['participant'] = ptpt_ids
        label_dict['trial'] = np.concat([np.arange(n_trials) for n_trials in trials_per_participant])
        
        if within is not None or covariates is not None:
            # within labels and covariates will be extracted from metadata
            # First generate metadata list
            if metadata_list is None:
                if not all([hasattr(item, 'metadata') for item in data]):
                    raise ValueError('Not all data objects contain metadata')
                else:
                    metadata_list = [item.metadata for item in data]
        # Extract columns of interest
        if within is not None:
            label_dict['within'] = pd.concat([md[within] for md in metadata_list])
        if covariates is not None:
            covariate_table = pd.concat([md[covariates] for md in metadata_list])

    # Enforce column order
    labels = pd.DataFrame(label_dict)
    labels = labels.reindex(columns=[c for c in ['between', 'participant', 'within', 'trial'] if c in labels.columns])
    # Detect single-subject analysis
    if labels.iloc[:, 0].nunique() == 1:
        labels = labels.drop(columns=labels.columns[0])
    
    # Get indicator of which factors stratify the data
    modeled = []
    for colname in labels.columns:
        if colname in ['between', 'within']:
            modeled.append(True)
        else:
            modeled.append(False)
    
    out = (datamat, labels, modeled)
    if covariates is not None:
        out += (covariate_table,)
    
    return out
