
import mne
import pyplsc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mne.stats.cluster_level import _find_clusters
import os, pathlib, lzma, pickle

from . import utils, viz

from pdb import set_trace

class BadStrArgError(Exception):
    def __init__(self, argname, provided, allowed):
        self.message = '%s is not a valid value for "%s". Must be one of %s' % (provided, argname, allowed)
        super().__init__(self.message)

def _check_str_arg(argname, provided, allowed):
    if provided not in allowed:
        raise BadStrArgError(argname, provided, allowed)

def fit_beh(data,
            covariates,
            design=None,
            between=None,
            within=None,
            participant=None,
            source_domain=None,
            source_freqs=None,
            boot_stat='score-covariate-corr',
            svd_method='lapack',
            random_state=None):
    """
    Fit behaviour PLS model.

    Parameters
    ----------
    data : MNE object or iterable of MNE objects
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers for epoched data (e.g., :class:`mne.Epochs`) and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers for averages over epochs (e.g., :class:`mne.Evoked`), and each observation will be a participant's average in a within-participants condition.
    covariates : ``np.ndarray`` | ``pd.DataFrame`` | iterable of ``str``
        Array or dataframe containing covariates, or an iterable of strings specifying the name(s) of the column(s) in ``design`` that contain the covariates.
    design : ``pd.DataFrame``, optional
        Design matrix containing indicators of experimental condition and/or covariates. The default is ``None``.
    between : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of between-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    within : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of within-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    participant : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of participant identity, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``. This is required only if there is a within-participants condition.
    source_domain : ``str``, optional
        If model is fit to source-space data, this argument specifies the domain of the source space. Must be one of:
        
        - ``'time'`` For output of :func:`mne.minimum_norm.apply_inverse`, :func:`mne.beamformer.apply_lcmv`, etc. This is assumed by default.
        - ``'freq'`` For output of :func:`mne.minimum_norm.apply_inverse_cov`, :func:`mne.beamformer.apply_dics`, etc.
        - ``'time-freq'`` For output of :func:`mne.minimum_norm.apply_dics_tfr_epochs, :func:`mne.beamformer.apply_dics_tfr_epochs`, etc.
    source_freqs : ``numpy.ndarray``, optional
        If model is fit to source-space data and source domain is time or time-frequency, this argument specifies the frequencies in the source data.
    boot_stat : ``str``, optional
        Specifies which statistic should be computed on each bootstrap iteration. The default is ``'score-covariate-corr'``. See :class:`pyplsc.PLSC` for details.
    svd_method : ``str``, optional
        The method of SVD decomposition. The default is ``'lapack'``. See :class:`pyplsc.PLSC` for details.
    random_state : ``int``, optional
        Random state for seeding the model. The default is None.

    Returns
    -------
    :class:`PLSC`
        PLSC object fit to the data.
    """
    
    template = Template(data,
                        source_domain=source_domain,
                        source_freqs=source_freqs)
    datamat = utils.get_datamat(data, template)
    model = pyplsc.PLSC(boot_stat,
                        svd_method,
                        random_state)
    model.fit(data=datamat,
              covariates=covariates,
              design=design,
              between=between,
              within=within,
              participant=participant)
    grouping = utils.get_grouping(between, within)
    return PLSC(template, model, grouping)

def fit_mc(data,
           design=None,
           between=None,
           within=None,
           participant=None,
           source_domain=None,
           source_freqs=None,
           effects='all',
           boot_stat='condwise-scores-centred',
           svd_method='lapack',
           random_state=None):
    """
    Fit mean-centred PLS model.

    Parameters
    ----------
    data : MNE object or iterable of MNE objects
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers for epoched data (e.g., :class:`mne.Epochs`) and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers for averages over epochs (e.g., :class:`mne.Evoked`), and each observation will be a participant's average in a within-participants condition. For source-space analysis, data will always be a list of source time courses.
    design : ``pd.DataFrame``, optional
        Design matrix containing indicators of experimental condition and/or covariates. The default is ``None``.
    between : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of between-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    within : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of within-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    participant : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of participant identity, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``. This is required only if there is a within-participants condition.
    source_domain : ``str``, optional
        If model is fit to source-space data, this argument specifies the domain of the source space. Must be one of:
        
        - ``'time'`` For output of :func:`mne.minimum_norm.apply_inverse`, :func:`mne.beamformer.apply_lcmv`, etc. This is assumed by default.
        - ``'freq'`` For output of :func:`mne.minimum_norm.apply_inverse_cov`, :func:`mne.beamformer.apply_dics`, etc.
        - ``'time-freq'`` For output of :func:`mne.minimum_norm.apply_dics_tfr_epochs, :func:`mne.beamformer.apply_dics_tfr_epochs`, etc.
    source_freqs : ``numpy.ndarray``, optional
        If model is fit to source-space data and source domain is time or time-frequency, this argument specifies the frequencies in the source data.
    boot_stat : ``str``, optional
        Specifies which statistic should be computed on each bootstrap iteration. The default is ``'score-covariate-corr'``. See :class:`pyplsc.BDA` for details.
    svd_method : ``str``, optional
        The method of SVD decomposition. The default is ``'lapack'``. See :class:`pyplsc.BDA` for details.
    random_state : ``int``, optional
        Random state for seeding the model. The default is None.

    Returns
    -------
    :class:`MCPLSC`
        MCPLSC object fit to the data.
    """
    
    template = Template(data,
                        source_domain=source_domain,
                        source_freqs=source_freqs)
    datamat = utils.get_datamat(data, template)
    model = pyplsc.BDA(boot_stat=boot_stat,
                       svd_method=svd_method,
                       random_state=random_state)
    model.fit(data=datamat,
              design=design,
              between=between,
              within=within,
              participant=participant,
              effects=effects)
    grouping = utils.get_grouping(between, within)
    return MCPLSC(template, model, grouping)

def fit_within_beh(data,
                   covariates,
                   within=None,
                   source_domain=None,
                   source_freqs=None,
                   boot_stat='score-covariate-corr',
                   svd_method='lapack',
                   random_state=None):
    """
    Fit within-participants behaviour PLS model.

    Parameters
    ----------
    data : MNE object or iterable of MNE objects
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers, and each observation will be a participant's average in a within-participants condition.
    covariates : iterable of ``str``
        An iterable of strings specifying the name(s) of the columns in the ``.metadata`` of each object in ``data`` that contain the covariates.
    within : ``str``, optional
        A string specifying the name of a column in the ``.metadata`` of each object in ``data`` that contains an indicator of within-participants condition. The default is ``None``, which does not stratify observations by within-participants condition.
    source_domain : ``str``, optional
        If model is fit to source-space data, this argument specifies the domain of the source space. Must be one of:
        
        - ``'time'`` For output of :func:`mne.minimum_norm.apply_inverse`, :func:`mne.beamformer.apply_lcmv`, etc. This is assumed by default.
        - ``'freq'`` For output of :func:`mne.minimum_norm.apply_inverse_cov`, :func:`mne.beamformer.apply_dics`, etc.
        - ``'time-freq'`` For output of :func:`mne.minimum_norm.apply_dics_tfr_epochs, :func:`mne.beamformer.apply_dics_tfr_epochs`, etc.
    source_freqs : ``numpy.ndarray``, optional
        If model is fit to source-space data and source domain is time or time-frequency, this argument specifies the frequencies in the source data.
    boot_stat : ``str``, optional
        Specifies which statistic should be computed on each bootstrap iteration. The default is ``'score-covariate-corr'``.
    svd_method : ``str``, optional
        The method of SVD decomposition. The default is ``'lapack'``.
    random_state : ``int``, optional
        Random state for seeding the model. The default is None.

    Returns
    -------
    :class:`PLSC`
        PLSC model fit to the data.
    """
    
    if not isinstance(data, list):
        data = [data]
    template = Template(data,
                        source_domain=source_domain,
                        source_freqs=source_freqs)
    datamat_list = [utils.get_datamat(ptpt, template) for ptpt in data]
    design_list = [ptpt.metadata for ptpt in data]
    model = pyplsc.WPLSC(boot_stat=boot_stat,
                         svd_method=svd_method,
                         random_state=random_state)
    model.fit(data=datamat_list,
              design=design_list,
              covariates=covariates,
              within=within)
    return PLSC(template, model, grouping='between')

class PLSC():
    """
    Container for PLSC models returned by :func:`fit_beh` and within-participants PLSC models returned by :func:`fit_within_beh`.
    
    Parameters
    ----------
    template : :class:`Template`
        A template object used for clustering and plotting.
    model : :class:`pyplsc.PLSC`
        A model that has been fit to some data.
    grouping : ``str``
        Specifies how data are stratified. Must be one of:
            
        - ``'between'``
        - ``'within'``
        - ``'both'``
        - ``'neither'``
    """
    def __init__(self, template, model, grouping):
        self.template = template #: :class:`Template`: A template containing storing channel positions etc. for plotting.
        self.model = model #: :class:`pyplsc.PLSC`: A PLSC model.
        self.grouping = grouping #: ``str``: Used to specify whether data are specified by within-participants condition, between-participants condition, both, or neither.
        self._clustering_done = False
        self.null_dist = None #: ``numpy.ndarray``: Array contain null distribution of singular values. Set by :meth:`permute`.
        self.clusters = None #: ``list``: A list of clusters per latent variable pair.
    def summary(self):
        """
        Summarize the model, including p values per latent variable pair if permutation has been done. See :meth:`permute`.

        Returns
        -------
        :class:`pandas.DataFrame`
            Data frame with one row per latent variable pair.
        
        Examples
        --------
        >>> res.summary()
        """
        return self.model.summary()
    def permute(self, n_perm=5000, store_null_dist=True, n_jobs=1, print_prog=True):
        """
        Perform permutation testing to assess the significance of the latent variables. p values become available after running this method through the :attr:`model.pvals_` attribute.

        Parameters
        ----------
        n_perm : int, optional
            Number of permutations t operform. The default is 5000.
        store_null_dist : bool, optional
            If ``True``, permutation samples will be saved and used for, e.g., scree plots. Default is ``True``.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.
        print_prog : bool, optional
            Specifies whether to display a progress bar. Default is ``True``.

        Returns
        -------
        None
        
        Notes
        -----
        p values are available through the :attr:`model.pvals_` attribute and can also be accessed using :meth:`summary`.

        Examples
        --------
        >>> res.permute(n_perm=1000, n_jobs=-1) # Use max parallel jobs
        >>> res.summary()
        >>> print(res.model.pvals_)
        """
        self.null_dist = self.model.permute(n_perm=n_perm,
                                            n_jobs=n_jobs,
                                            print_prog=print_prog,
                                            return_null_dist=store_null_dist)
    def bootstrap(self, n_boot=5000, confint_level=0.95, alignment_method='rotate-design-sals', return_boot_stat_dist=False, n_jobs=1, print_prog=True):
        """
        Perform (stratified) bootstrap resampling to assess the reliability of the data saliences.

        Parameters
        ----------
        n_boot : int, optional
            Number of bootstrap resamples to compute. The default is 5000.
        confint_level : float, optional
            The confidence level of the quantile-based confidence intervals to compute. The default is 0.95.
        alignment_method : string, optional
            Method to be used for aligning recomputed data saliences with original data saliences. Must be one of:
            
            - ``'rotate-design-sals'`` (default): Find the rotation that solves the orthogonal procrustes problem to align the recomputed and original design saliences, then apply this to the recomputed data saliences. This is the what is computed in the original Matlab version of PLS.
            - ``'rotate-data-sals'``: Find the rotation that solves the orthogonal procrustes problem to align the recomputed and original data saliences, then apply this to the recomputed data saliences.
            - ``'flip-design-sals'``: Find the set of sign flips that ensures the inner product of the recomputed and original design saliences are positive, then apply these sign flips to the recomputed data saliences.
            - ``'flip-data-sals'``: Find the set of sign flips that ensures the inner product of the recomputed and original data saliences are positive, then apply these sign flips to the recomputed data saliences.
            - ``'none'``: Perform no alignment.
        return_boot_stat_dist : bool, optional
            If ``True``, distribution of ``boot_stat`` from resampling is returned. This is the distribution used to compute quantile-based confidence intervals. Default is ``True``.
        n_jobs : int, optional
            Number of parallel jobs to deploy to compute permutations. -1 automatically deploys the maximum number of jobs. The default is 1.
        print_prog : bool, optional
            Specifies whether to display a progress bar. Default is ``True``.

        Returns
        -------
        :class:`numpy.ndarray`
            If `return_boot_dist` is true, returns the bootstrap distribution of the statistic named by :attr:`model.boot_stat`

        Examples
        --------
        >>> res.bootstrap(1000, n_jobs=-1)
        >>> print(res.model.boot_stat_ci[..., 0]) # Print CI of boot_stat for first LV
        """
        self.model.bootstrap(n_boot=n_boot,
                             confint_level=confint_level,
                             alignment_method=alignment_method,
                             return_boot_stat_dist=return_boot_stat_dist,
                             n_jobs=n_jobs,
                             print_prog=print_prog)
    def brain_sals_to_mne(self, lv_idx, which='saliences'):
        _check_str_arg('which', which,
                       ['saliences', 'z-scores'])
        if which == 'saliences':
            data = self.model.data_sals_[:, lv_idx]
        elif which == 'z-scores':
            data = self.model.data_sals_z_[:, lv_idx]
        data = data.reshape(self.template.shape)
        if self.template.space == 'sensor':
            info = self.template.info
            if self.template.domain == 'time':
                out_obj = mne.EvokedArray(data=data,
                                          info=info,
                                          tmin=self.template.times[0])
            elif self.template.domain == 'freq':
                out_obj = mne.time_frequency.SpectrumArray(data=data,
                                                           info=info,
                                                           freqs=self.template.freqs)
            elif self.template.domain == 'time-freq':
                out_obj = mne.time_frequency.AverageTFRArray(data=data,
                                                             info=info,
                                                             times=self.template.times,
                                                             freqs=self.template.freqs)
        elif self.template.space == 'source':
            if self.template.source_type == 'surface':
                class_constructor = mne.SourceEstimate
            elif self.template.source_type == 'volume':
                class_constructor = mne.VolSourceEstimate
            kwargs = dict(vertices=self.template.vertices,
                          tmin=self.template.times[0],
                          tstep=self.template.tstep)
            if self.template.domain in ('time', 'freq'):
                # Single STC
                out_obj = class_constructor(data=data, **kwargs)
            elif self.template.domain == 'time-freq':
                # List of frequency-specific STCs
                out_obj = list()
                for freq_idx in range(len(self.template.freqs)):
                    curr_data = data[:, freq_idx, :].squeeze()
                    freq_stc = class_constructor(data=curr_data, **kwargs)
                    out_obj.append(freq_stc)
        return out_obj
    def add_source_info(self, src=None, mri=None, subjects_dir=None, freqs=None):
        """
        Add information about source space for clustering and plotting.

        Parameters
        ----------
        src : :class:`mne.SourceSpaces`, optional
            Source spaces corresponding to the source time courses. The default is None.
        mri : Niimg-like object, optional
            Structural data, e.g., path to T1 scan file. The default is None.
        subjects_dir : path-like, optional
            Freesurfer subjects directory. The default is None.

        Returns
        -------
        None
        """
        if not self.template.space == 'source':
            raise ValueError('Data is not in source space.')
        if src is not None:
            if src.kind != self.template.source_type:
                raise ValueError('src.kind is %s but data is in %s source space' % (src.kind, self.template.source_type))
            self.template.src = src
        if mri is not None:
            self.template.mri = mri
        if subjects_dir is not None:
            self.template.subjects_dir = subjects_dir
        if freqs is not None:
            if self.template.domain == 'time':
                raise ValueError('Attempting to add frequencies, but data is time-domain')
            self.template.freqs = np.array(freqs)
            self.template.domain = 'freq'
    def add_adjacency(self, all_space_adjacent='auto', montage_name=None, spatial_adjacency=None):
        """
        Add adjacency matrix for clustering.

        Parameters
        ----------
        all_space_adjacent : bool, optional
            Specifies whether all spatial locations (e.g. channels, vertices) should be treated as adjacent to each other for the purposes of clustering. This is useful when doing ERP analyses, where strong loadings at non-adjacent channels would be considered part of the same component. It can also be used to examine distributed patterns of (de)synchronization for frequency-domain analyses. The default is ``'auto'``, which is ``True`` for ERP analysis (inferred based on :attr:`template.datatype`) and ``False`` for all other analyses.
        montage_name : str, optional
            Name of montage passed to ``mne.channels.read_ch_adjacency``. The default is ``None``, which uses ``mne.channels.find_ch_adjacency`` to get channel adjacency.

        Returns
        -------
        None.
            None. Adds an ``adjacency`` attribute to :attr:`template` which indicates which channels, times, and frequencies (as applicable) are adjacent for clustering.
        """
        if all_space_adjacent == 'auto':
            if self.template.datatype == 'epo':
                all_space_adjacent = True
                print('Defaulting to all channels adjacent for ERP/ERF analysis')
            else:
                all_space_adjacent = False
        if self.template.space == 'sensor':
            if all_space_adjacent:
                spatial_adj = np.ones((self.template.info['nchan'],)*2)
            else:
                if montage_name is None:
                    ch_types = set(self.template.info.get_channel_types())
                    if len(ch_types) > 1:
                        raise ValueError('Multiple channel types present in data: %s. Adjacency could not be computed' % ch_types)
                    ch_type = ch_types.pop() # One-element set
                    spatial_adj, _ = mne.channels.find_ch_adjacency(self.template.info, ch_type)
                else:
                    spatial_adj, _ = mne.channels.read_ch_adjacency(montage_name)
        elif self.template.space == 'source':
            if self.template.src is None:
                raise ValueError('Source space must be specified to compute spatial adjacency. See add_source_info()')
            # TODO: other options for surface source spaces
            if all_space_adjacent:
                n_vert = sum(len(ss['vertno']) for ss in self.template.src)
                spatial_adj = np.ones((n_vert,)*2)
            else:
                spatial_adj = mne.spatial_src_adjacency(self.template.src)
        dim_adjs = (spatial_adj,) + self.template.shape[1:]
        self.template.adjacency = mne.stats.combine_adjacency(*dim_adjs)
    def cluster(self, which='auto', threshold=None, signed='auto'):
        """
        Identify clusters of adjacent saliences above some threshold.

        Parameters
        ----------
        which : str, optional
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be used for clustering. The default is ``'auto'``, which uses z-scores if they are available and otherwise falls back to raw saliences.
        threshold : float | callable, optional
            Saliences must be above this threshold to be part of a cluster. The default is ``None``, which uses the mean salience if ``which='saliences'`` and a value of 2 if ``which='z-scores'``.
        signed : bool, optional
            If ``True``, each cluster will contain only positive or only negative saliences. If ``False``, clusters can contain both positive and negative saliences. In ERP analysis, both positive and negative saliences could be considered part of the same component. The default is ``'auto'``, which is ``False`` for ERP analysis and ``True`` for other analyses.

        Returns
        -------
        None
            None. Adds the attribute :attr:`clusters`.
        """
        _check_str_arg('which', which, ('auto', 'saliences', 'z-scores'))
        if 'adjacency' not in dir(self.template):
            raise ValueError('Adjacency must be added with .add_adjacency() before clustering can be done')
        # Auto-determine which data to cluster
        if which == 'auto':
            if self.model._boot_done:
                which = 'z-scores'
            else:
                which = 'saliences'
            print('Clustering %s' % which)
        # Validate data to cluster
        if which == 'z-scores':
            if not self.model._boot_done:
                raise ValueError('Bootstrap resampling must be done to use z scores for clustering.')
            data = self.model.data_sals_z_
            if threshold is None:
                # Conventional threshold, z score of 2
                threshold = 2
        elif which == 'saliences':
            data = self.model.data_sals_
            if threshold is None:
                # Average salience
                threshold = np.mean
        # Compute abs---ends up being used even for signed clustering
        absdata = np.abs(data)
        if callable(threshold):
            threshold = np.apply_along_axis(func1d=threshold,
                                            axis=0,
                                            arr=absdata)
        # In case threshold is a scalar, repeat per LV
        try:
            len(threshold)
        except:
            threshold = [threshold]*self.model.n_sv_
        
        if signed == 'auto':
            if self.template.domain == 'time':
                print('Defaulting to unsigned clustering')
                signed = False
            else:
                signed = True
        if not signed:
            data = absdata
        
        clusters = []
        for lv_idx in range(self.model.rank_):
            # Separate clustering for positive and negative
            print('Computing clusters for lv_idx %s...' % lv_idx)
            curr_thresh = threshold[lv_idx]
            n_above_thresh = np.sum(absdata[:, lv_idx] > curr_thresh)
            idxs, sums = _find_clusters(
                data[:, lv_idx],
                tail=0,
                threshold=curr_thresh,
                adjacency=self.template.adjacency)
            # Sort largest to smallest
            idxs.sort(key=len, reverse=True)
            print('%s clusters' % len(idxs))
            # Get peaks of each cluster
            peaks = []
            for clust_idx in idxs:
                # Get linear index of max
                peak_flat = clust_idx[absdata[clust_idx, lv_idx].argmax()]
                # Get coords of max
                peak_coords = np.unravel_index(peak_flat, self.template.shape)
                peaks.append((peak_coords, peak_flat))
            clusters.append({
                'info': {
                    'which': which,
                    'threshold': curr_thresh,
                    'n_above_thresh': n_above_thresh
                },
                'clusters': [{'idx': idx, 'peak_coords': peak_coords, 'peak_flat': peak_flat} for idx, (peak_coords, peak_flat) in zip(idxs, peaks)]
            })
        self.clusters = clusters
        self._clustering_done = True
    def _get_cluster(self, lv_idx, cluster_idx, return_data=True, which='auto'):
        _check_str_arg('which', which,
                       ('auto', 'data', 'saliences', 'z-scores'))
        lv_clusters = self.clusters[lv_idx]
        info = lv_clusters['info']
        # Create copy of cluster and add mask
        cluster = lv_clusters['clusters'][cluster_idx].copy()
        # Go from linear indices to ndarray mask
        mask = np.zeros(self.template.shape, dtype=np.bool)
        mask.flat[cluster['idx']] = True
        cluster['mask'] = mask
        out = cluster, info
        if return_data:
            if which == 'auto':
                # Default to data used for clustering
                which = info['which']
            if which == 'data':
                data = self.model.data_[:, lv_idx]
            elif which == 'saliences':
                data = self.model.data_sals_[:, lv_idx]
            elif which == 'z-scores':
                data = self.model.data_sals_z_[:, lv_idx]
            reshaped = data.copy().reshape(self.template.shape)
            out += (reshaped,)
        return out
    def cluster_to_stc(self, lv_idx, cluster_idx, which='auto', mask_val=0):
        """
        SUMMARY.

        Parameters
        ----------
        lv_idx : TYPE
            DESCRIPTION.
        cluster_idx : TYPE
            DESCRIPTION.
        mask_val : TYPE, optional
            DESCRIPTION. The default is 0.

        Returns
        -------
        TYPE
            DESCRIPTION.

        Raises
        ------
        ValueError
            DESCRIPTION.
        """
        # TODO: document
        if self.template.space != 'source':
            raise ValueError('Data must be in source space')
        cluster, _, data = self._get_cluster(lv_idx, cluster_idx, return_data=True, which=which)
        data[~cluster['mask']] = mask_val
        if self.template.domain == 'freq':
            # Placeholder values, like MNE does
            tmin = 0
            tstep = 1
        else:
            times = self.template.times * 1000 # s to ms
            tmin = times[0]
            tstep = np.diff(times)[0]
        if self.template.domain == 'time-freq':
            # Average over frequencies in cluster
            masked = np.ma.masked_array(data=data,
                                        mask=~cluster['mask'])
            data = np.array(masked.mean(axis=1))
        if self.template.source_type == 'surface':
            constructor = mne.SourceEstimate
        elif self.template.source_type == 'volume':
            constructor = mne.VolSourceEstimate
        stc = constructor(data=data,
                          vertices=self.template.vertices,
                          tmin=tmin,
                          tstep=tstep,
                          subject=self.template.subject) # TODO: does this make sense in general? Probably not. Subject should be fsaverage for multi-subject analysis but the participant's own ID for single-subject
        return stc
    def cluster_to_volume(self, lv_idx, cluster_idx, mask_val=0):
        """
        SUMMARY.

        Parameters
        ----------
        lv_idx : TYPE
            DESCRIPTION.
        cluster_idx : TYPE
            DESCRIPTION.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        # TODO: document
        if self.template.space == 'source':
            if self.template.source_type == 'volume':
                stc = self.cluster_to_stc(lv_idx, cluster_idx, mask_val)
                img = stc.as_volume(src=self.template.src)
                return img
            else:
                raise ValueError('Model must be in volume source space')
        else:
            raise ValueError('Model is not in source space')
    def get_cluster_sizes(self, lv_idx, size_measure='pct-strong'):
        """
        Get sizes of clusters.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which the plot should be generated.
        size_measure : str, optional
            Specifies how cluster size should be measured. Must be one of:
            
            - ``'pct-strong'`` (default): cluster sizes are as a percentage of the strong saliences (strong meaning above the threshold used in :meth:`cluster()`)
            - ``'pct-total'``: cluster sizes are as a percentage of the total number of saliences per singular vector.
            - ``'absolute'``: cluster sizes are in absolute number of saliences.

        Returns
        -------
        np.ndarray
            The sizes of each cluster, sorted from largest to smallest.
        """
        if not self._clustering_done:
            raise ValueError('Clustering has not been performed')
        _check_str_arg('size_measure', size_measure,
                       ('pct-strong', 'pct-total', 'absolute'))
        cluster_set = self.clusters[lv_idx]
        abs_sizes = np.array([len(c['idx']) for c in cluster_set['clusters']])
        if size_measure == 'absolute':
            sizes = abs_sizes
        elif size_measure == 'pct-strong':
            sizes = 100*abs_sizes/cluster_set['info']['n_above_thresh']
        elif size_measure == 'pct-total':
            sizes = 100*abs_sizes/self.template.size
        return sizes
    def get_cluster_data(self, lv_idx=None, cluster_idx=None):
        """
        Extract the data for each observation in a given cluster or set of clusters as a dataframe. Note that this extracts the actual brain data rather than the brain saliences.

        Parameters
        ----------
        lv_idx : indexer, optional
            Index of the latent variable(s) for which to extract data. The default is None.
        cluster_idx : TYPE, optional
            DESCRIPTION. The default is None.
        
        Returns
        -------
        pd.DataFrame
            A data frame containing .
        """
        
        if not self._clustering_done:
            raise ValueError('Clustering needs to be done first via the cluster() method.')
        if lv_idx is None:
            lv_idx = list(range(self.model.n_sv_))
        else:
            try:
                len(lv_idx)
            except:
                lv_idx = [lv_idx]
        dfs = []
        for curr_lv_idx in lv_idx:
            lv_clusters = self.clusters[curr_lv_idx]['clusters']
            if cluster_idx is None:
                cluster_idx = list(range(len(lv_clusters)))
            else:
                try:
                    len(cluster_idx)
                except:
                    cluster_idx = [cluster_idx]
            for curr_cluster_idx in cluster_idx:
                # Set up sub-dataframe for this cluster
                df = self.model.get_design_matrix()
                df['lv_idx'] = curr_lv_idx
                df['cluster_idx'] = curr_cluster_idx
                curr_cluster = lv_clusters[curr_cluster_idx]
                # Take average within cluster
                df['cluster_mean'] = self.model.data_[:, curr_cluster['idx']].mean(axis=1)
                # Get data at cluster peak
                df['cluster_peak'] = self.model.data_[:, curr_cluster['peak_flat']]
                dfs.append(df)
        # TODO: remove columns based on grouping (e.g., no redundant "between" column if it's never applicable)
        return pd.concat(dfs)
        
    def plot_scree(self, which='pct-variance', null_percentile=95, ax=None):
        """
        Generate a scree plot of singular values, possibly along with their null distributions.

        Parameters
        ----------
        which : str, optional
            Specifies whether the raw singular values (``'singular-vals'``) or percent variance explained (``'pct-variance'``) should be plotted. The default is ``'pct-variance'``.
        null_dist : numpy.ndarray, optional
            Null distribution of singular values to display, as returned by :attr:`model.permute()`. The default is ``None``, which plots no null distribution.
        null_percentile : float, optional
            If provided, the null distribution of each singular value is displayed as a vertical line from the minimum to a percentile specified by this argument. The default is ``95``.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Notes
        -----
        If ``which`` is ``'pct-variance'``, the singular values in the null distribution are squared and divided by the sum of the squared observed singular values.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """
        _check_str_arg('which', which,
                       ('pct-variance', 'singular-vals'))
        viz.scree(singular_vals=self.model.singular_vals_,
                  which=which,
                  rank=self.model.rank_,
                  null_dist=self.null_dist,
                  null_percentile=null_percentile,
                  ax=ax)
    def plot_scores(self, lv_idx, ax=None):
        """
        Create a scatterplot of data scores against design scores.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which the plot should be generated.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """
        
        df = self.model.get_scores_frame(lv_idx)
        f, ax = viz.score_scatterplot(df, self.grouping, ax=ax)
        return f, ax
    def plot_boot_stat(self, lv_idx, with_ci=True, ax=None):
        """
        Visualize :attr:`model.boot_stat` with a barplot.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which the plot should be generated.
        with_ci : bool, optional
            Specifies whether to show confidence interval error bars, if bootstrapping has been done. Ignored if bootstrapping has not been done. Default is ``True``.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """        
        df = self.model.get_boot_stat_frame(lv_idx)
        out = viz.boot_stat_barplot(df=df,
                                    boot_stat=self.model.boot_stat,
                                    grouping=self.grouping,
                                    with_ci=self.model._boot_done and with_ci,
                                    ax=ax)
        return out
    def plot_brain_sals(self, lv_idx, which='saliences', ax=None):
        """
        Plot of brain saliences.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which the plot should be generated.
        which : str, optional
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be plotted. The default is `'saliences'`.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """
        _check_str_arg('which', which,
                       ('saliences', 'z-scores'))
        if which == 'z-scores':
            data = self.model.data_sals_z_[:, lv_idx]
            label = 'z score'
            avg_label = 'Mean z score'
        elif which == 'saliences':
            data = self.model.data_sals_[:, lv_idx]
            label = 'Salience'
            avg_label = 'Mean salience'
        data = data.reshape(self.template.shape)
        if self.template.domain == 'time-freq':
            # Show average over spatial domain for both sensor and source data
            tf_data = data.mean(axis=0)
            f, ax = viz.plot_labeled_raster(template=self.template,
                                            data=tf_data,
                                            xdim='time',
                                            ydim='freq',
                                            vlabel=avg_label,
                                            ax=ax)
        else:
            if self.template.space == 'sensor':
                # Line plot with spatial colours
                if self.template.domain == 'time':
                    xdata = self.template.times
                    xlabel = 'Time (s)'
                elif self.template.domain == 'freq':
                    xdata = self.template.freqs
                    xlabel = 'Frequecy (Hz)'
                if self._clustering_done:
                    ythresh = self.clusters[lv_idx]['info']['threshold']
                else:
                    ythresh = None
                f, ax = viz.channel_lineplot(x=xdata,
                                             ch_y=data,
                                             info=self.template.info,
                                             ax=ax,
                                             xlabel=xlabel,
                                             ylabel=label,
                                             ythresh=ythresh)
            elif self.template.space == 'source':
                if self.template.source_type == 'surface':
                    # Raster is the best we can do for now
                    ydim, xdim = self.template.dimnames
                    f, ax = viz.plot_labeled_raster(template=self.template,
                                                    data=data,
                                                    xdim=xdim,
                                                    ydim=ydim,
                                                    vlabel=label,
                                                    ax=ax)
                elif self.template.source_type == 'volume':
                    if self.template.src is None:
                        print('Plotting raster image. To view a 4D image, add source info via the add_source_info() method')
                        ydim, xdim = self.template.dimnames
                        f, ax = viz.plot_labeled_raster(template=self.template,
                                                        data=data,
                                                        xdim=xdim,
                                                        ydim=ydim,
                                                        vlabel=label,
                                                        ax=ax)
                    else:
                        stc = self.brain_sals_to_mne(lv_idx=lv_idx,
                                                     which=which)
                        img = stc.as_volume(src=self.template.src)
                        if self.template.domain == 'time':
                            xdata = self.template.times
                            xlabel = 'Time (s)'
                        elif self.template.domain == 'freq':
                            xdata = self.template.freqs
                            xlabel = 'Frequecy (Hz)'
                        f, ax = viz.plot_niimg_4d(img=img, 
                                                  xdata=xdata,
                                                  xlabel=xlabel,
                                                  vlabel=label,
                                                  ax=ax)
        return f, ax
            
    def plot_lv(self, lv_idx, which='saliences'):
        """
        Create a two-panel summary plot of a latent variable pair. The left panel displays the value of :attr:`boot_stat` while the right panel displays the brain saliences.

        Parameters
        ----------
        lv_idx : indexer
            Index of latent variable pair(s) for which the plot should be generated.
        which : str, optional
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be plotted in the right panel. The default is `'saliences'`.

        Returns
        -------
        f, ax
            Figure and axes containing plots.
        """
        
        f, ax = plt.subplots(nrows=1, ncols=2,
                             width_ratios=[3, 2],
                             layout='constrained')
        self.plot_brain_sals(lv_idx, ax=ax[0], which=which)
        self.plot_boot_stat(lv_idx, ax=ax[1])
        return f, ax
    def plot_cluster_sizes(self, lv_idx, size_measure='pct-strong', n_clust=None, ax=None):
        """
        Create a plot of cluster sizes from largest to smallest.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which cluster sizes should be plotted.
        size_measure: str, optional
            Specifies how cluster size should be measured. See :meth:`get_cluster_sizes`. The default is `'pct-strong'`.
        n_clust : ``int``, optional
            Number of clusters, starting from largest, to plot. Default is ``None``, which displays all clusters.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """
        cluster_sizes = self.get_cluster_sizes(lv_idx=lv_idx,
                                               size_measure=size_measure)
        if n_clust is not None:
            cluster_sizes = cluster_sizes[:n_clust]
        out = viz.plot_cluster_sizes(cluster_sizes=cluster_sizes,
                                     size_measure=size_measure,
                                     ax=ax)
        return out
    def plot_cluster_nonspatial(self, lv_idx, cluster_idx, highlight='none', plot_type='auto', ax=None):
        """
        SUMMARY.

        Parameters
        ----------
        lv_idx : TYPE
            DESCRIPTION.
        cluster_idx : TYPE
            DESCRIPTION.
        annotate : str
            What to annotate.
        plot_type : TYPE, optional
            DESCRIPTION. The default is 'auto'.
        ax : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        None
        """
        if ax is None:
            f, ax = plt.subplots()
        if plot_type == 'auto':
            if self.template.datatype in ['epo', 'spec']:
                plot_type = 'butterfly'
            elif self.template.domain == 'time-freq':
                plot_type = 'raster'
            elif self.template.datatype in ['surf-stc', 'vol-stc']:
                plot_type = 'distribution'
        # Check for mismatch between data type and plot type
        if self.template.datatype in ['epo', 'spec']:
            valid_plot_types = ['butterfly', 'raster', 'distribution']
        else:
            valid_plot_types = ['raster', 'distribution']
        if plot_type not in valid_plot_types:
            raise ValueError('%s is not a valid plot type for this data. Valid options are %s.' % (plot_type, valid_plot_types))
        cluster, info, data = self._get_cluster(lv_idx, cluster_idx, return_data=True, which='auto')
        if plot_type == 'butterfly':
            out = viz.plot_cluster_butterfly(data=data,
                                             template=self.template,
                                             cluster=cluster,
                                             which=info['which'],
                                             ythresh=info['threshold'],
                                             highlight=highlight,
                                             ax=ax)
        elif plot_type == 'raster':
            out = viz.plot_cluster_raster_data(data=data,
                                               template=self.template,
                                               cluster=cluster,
                                               which=info['which'],
                                               highlight=highlight,
                                               ax=ax)
        elif plot_type == 'distribution':
            out = viz.plot_cluster_distribution(self.template,
                                                cluster=cluster,
                                                highlight=highlight,
                                                ax=ax)
        return out
      
    def plot_cluster_spatial(self, lv_idx, cluster_idx, highlight='peak', ax=None):
        """
        Plot cluster across spatial dimension of data (sensors or sources).

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair.
        cluster_idx : int
            Index of cluster.
        highlight : str, optional
            See ``highlight`` arg to :meth:`plot_clusters`
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is ``None``, which generates a new figure.

        Returns
        -------
        TYPE
            DESCRIPTION.
        """
        
        if self.template.source_type == 'surface':
            # Interactive surface plot
            # Determine whether cluster peak is in left or right hemisphere
            cluster, cluster_info = self._get_cluster(lv_idx, cluster_idx, return_data=False)
            if self.template.domain == 'time':
                vert_peak, time_peak = cluster['peak_coords']
                initial_time = self.template.times[time_peak]
            elif self.template.domain == 'freq':
                vert_peak, freq_peak = cluster['peak_coords']
                initial_time = self.template.freqs[freq_peak]
            elif self.template.domain == 'time-freq':
                vert_peak, freq_peak, time_peak = cluster['peak_coords']
                initial_time = self.template.times[time_peak]
            if vert_peak < self.template.vertices[0].size:
                hemi = 'lh'
            else:
                hemi = 'rh'
            # Convert to STC object for plotting
            stc = self.cluster_to_stc(lv_idx, cluster_idx)
            # Set colour limits from threshold to max
            if self.template.domain == 'time-freq':
                # We've averaged over frequency when converting to stc
                peak_vert, _, peak_time = cluster['peak_coords']
                peak_coords = (peak_vert, peak_time)
            else:
                peak_coords = cluster['peak_coords']
            cmax = np.abs(stc.data[peak_coords])
            cmin = cluster_info['threshold']
            cmid = cmin
            clim = (cmin, cmid, cmax)
            # Generate interactive plot
            out = stc.plot(subjects_dir=self.template.subjects_dir,
                           hemi=hemi,
                           time_viewer=True,
                           initial_time=initial_time,
                           clim={'kind': 'value',
                                 'pos_lims': clim})
        else:
            cluster, info, data = self._get_cluster(lv_idx, cluster_idx)
            out = viz.plot_cluster_spatial(data=data,
                                           template=self.template,
                                           cluster=cluster,
                                           cluster_info=info,
                                           highlight=highlight,
                                           ax=ax)
        return out
    def plot_cluster(self, lv_idx, cluster_idx, size_measure='pct-strong', highlight='peak', plot_type='auto'):
        """
        Visualize both the spatial and non-spatial dimensions of a cluster of strong loadings. 

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair of the cluster to be plotted.
        cluster_idx : int
            Index of cluster to display.
        size_measure : str, optional
            Specifies the size measure to use when comparing cluster sizes to ``min_size``. See :meth:`get_cluster_sizes`. The default is ``'pct-strong'``. Ignored if ``cluster_idx`` is specified.
        highlight : str, optional
            Specifies how to represent the cluster spatially. Must be one of:
                
            - ``'peak'`` (default): Plots the spatial data at the peak across the non-spatial dimensions.
            - ``'extent'``:  Plots the spatial data averaged over the cluster extent over the non-spatial dimensions.
        plot_type : str, optional
            Specifies how to plot the non-spatial data. Must be one of:
                
            - ``'auto'`` (default): Makes a sensible choice given the datatype provided.
            - ``'butterfly'``: Creates a butterfly plot (coloured line plot per channel)
            - ``''

        Returns
        -------
        None
        """
        # TODO: add option to add a higher threshold for ease of visualization
        _check_str_arg('highlight', highlight, ['peak', 'extent']) # Can't be none here, even though it can be non for the non-spatial plot
        if self.template.datatype == 'surf-stc':
            raise ValueError('Spatial and non-spatial cluster visualizations cannot be shown in the same figure. Call .plot_cluster_nonspatial() and .plot_cluster_spatial() separately.')
        else:
            f, ax = plt.subplots(ncols=2, layout='constrained')
            # Left axis: visualize non-spatial dimension(s)
            self.plot_cluster_nonspatial(lv_idx=lv_idx,
                                         cluster_idx=cluster_idx,
                                         plot_type=plot_type,
                                         highlight=highlight,
                                         ax=ax[0])
            # Right axis: visualize spatial dimension
            self.plot_cluster_spatial(lv_idx=lv_idx,
                                      cluster_idx=cluster_idx,
                                      highlight=highlight,
                                      ax=ax[1])
        return f, ax
    def save(self, path):
        """
        Save a model to .xz using the LZMA algorithm. This is a thin wrapper around python's ``lzma`` library.

        Parameters
        ----------
        path : ``str``
            Path to model. If no extension is included, .xz will be added.

        Returns
        -------
        None
        """
        
        path = pathlib.Path(path)
        file, ext = os.path.splitext(path)
        if ext == '':
            path = path.with_suffix('.xz')
            basename = os.path.basename(path)
            print('Saving to %s' % basename)
        with lzma.open(path, "wb") as f:
            pickle.dump(self, f)

def load(path):
    """
    Load a model from .xz. This is a thin wrapper around Python's ``lzma`` library.

    Parameters
    ----------
    path : ``str``
        Path to model. If no extension is included, .xz will be added.

    Returns
    -------
    model
        The loaded model, of the same .
    """
    
    path = pathlib.Path(path)
    file, ext = os.path.splitext(path)
    if ext == '':
        path = path.with_suffix('.xz')
        basename = os.path.basename(path)
        print('Loading %s' % basename)
    with lzma.open(path, 'rb') as f:
        model = pickle.load(f)
    return model

class MCPLSC(PLSC):
    """
    Container for mean-centred PLSC models returned by :func:`fit_mc`.
    
    Parameters
    ----------
    template : :class:`Template`
        A template object used for clustering and plotting.
    model : :class:`pyplsc.BDA`
        A model that has been fit to some data.
    grouping : ``str``
        Specifies how data are stratified. Must be one of:
            
        - ``'between'``
        - ``'within'``
        - ``'both'``
    """
    def get_marginal_brain_scores(self, lv_idx, margin, average=True):
        """
        Compute marginal brain scores per condition across a specified margin. This generalizes the notion temporal brain scores from the original Matlab PLS.

        Parameters
        ----------
        lv_idx : int
            The index of the latent variable pair for which marginal brain scores should be computed.
        margin : str
            The margin across which brain scores should be computed. Must be one of:
            
            - ``'chan'``: channel
            - ``'time'``: time (computes temporal brain scores)
            - ``'freq'``: frequency
            - ``'time-freq'``: both time and frequency
        average : bool
            Specifies whether condition-wise averages should be computed.

        Notes
        -----
        Use :meth:`model.data_sal_labels_` to determine which elements of the output list correspond to which experimental conditions.        

        Returns
        -------
        list of numpy.ndarray
            Marginal brain scores per condition.
        
        Examples
        --------
        >>> scores = mod.get_marginal_brain_scores(lv_idx=0, margin='time') # Temporal brain scores
        >>> scores = mod.get_marginal_brain_scores(lv_idx=0, margin='freq') # Spectral brain scores
        """
        _check_str_arg('margin', margin,
                       ('chan', 'time', 'freq', 'time-freq'))
        if margin != 'chan':
            if margin == 'time':
                allowed_datatypes = ('epo', 'tfr', 'surf-stc', 'vol-stc')
            elif margin == 'freq':
                allowed_datatypes = ('spec', 'tfr')
            elif margin == 'time-freq':
                allowed_datatypes = ('tfr',)
                assert self.template.datatype == 'tfr'
            if self.template.datatype not in allowed_datatypes:
                raise ValueError('Marginal brain scores over margin "%s" can only be computed for datatypes %s' % (margin, allowed_datatypes))
        # ('chan' is allowed for all)

        # Compute hadamard products
        loadings = self.model.data_sals_[:, lv_idx]
        hadamards = self.model.data_ * loadings
        if average:
            hadamards = pyplsc.utils.get_groupwise_means(
                data=hadamards,
                group_idx=self.model.stratifier_)
        # Reshape
        hadamards = [h.reshape(self.template.shape) for h in hadamards]
        # Identify axes to average over
        non_margin_axes = utils.get_non_margin_axes(margin, self.template.datatype)
        # Compute marginal scores
        scores = [h.mean(axis=non_margin_axes) for h in hadamards]
        return scores
    def plot_marginal_brain_scores(self, lv_idx, margin):
        """
        Plot marginal brain scores.

        Parameters
        ----------
        lv_idx : int
            The index of the latent variable pair for which marginal brain scores should be computed.
        margin : str
            The margin across which brain scores should be computed. See :meth:`get_marginal_brain_scores`.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        
        Examples
        --------
        >>> mod.plot_marginal_brain_scores(lv_idx=0, margin='time')
        >>> mod.plot_marginal_brain_scores(lv_idx=0, margin='freq')
        """
        scores = self.get_marginal_brain_scores(lv_idx=lv_idx,
                                                margin=margin,
                                                average=True)
        labels = self.model.design_sal_labels_
        out = viz.plot_marginal_brain_scores(scores=scores,
                                             margin=margin,
                                             labels=labels,
                                             template=self.template,
                                             grouping=self.grouping)
        return out

class Template():
    """
    Template containing channels, source info, times, frequencies, etc. associated with the data. This is used for clustering and plotting.
    """
    def __init__(self, obj, source_domain=None, source_freqs=None):
        # Document attributes
        self.src = None #: :class:`mne.SourceSpaces`: Source spaces of data, if applicable.
        self.mri = None #: Niimg-like: Structural MRI data, if applicable.
        self.subjects_dir = None #: path-like: Freesurfer subjects directory, if applicable.
        self.datatype = None #: ``str``: Specifies the type of the data.
        self.source_type = None #: ``str``: Differentiates between surface and volume sources
        self.shape = None #: ``tuple``: Specifies the shape of the data array.
        self.size = None #: ``int``: Size of data.
        self.times = None #: ``numpy.ndarray``: Times, copied from data
        self.freqs = None #: ``numpy.ndarray``: Frequencies, copied from data
        self.subject = None #: ``str``: Freesurfer subject name, copied from data
        self.domain = None #: ``str``: Specifies whether data is time-domain, frequency-domain, or time-frequency.
        self.tstep = None # TODO: document
        _check_str_arg('domain', source_domain,
                       (None, 'time', 'freq', 'time-freq'))
        # Infer datatype
        if isinstance(obj, list):
            inst = obj[0]
            if isinstance(inst, list):
                inst = inst[0]
                # List of lists implies source-space time-frequency
                self.space = 'source' # Redundant with later code but no harm
                self.domain = 'time-freq'
                if source_freqs is None:
                    raise ValueError('Frequencies must be specified for time-frequency data in source space')
        else:
            inst = obj
        attrs = dir(inst)
        if 'times' not in attrs:
            self.space = 'sensor'
            self.datatype = 'spec'
            self.domain = 'freq'
        else:
            if 'vertices' in attrs:
                self.space = 'source'
                if 'as_volume' in attrs:
                    self.source_type = 'volume'
                    self.datatype = 'vol-stc'
                else:
                    self.source_type = 'surface'
                    self.datatype = 'surf-stc'
                if self.domain is None:
                    if source_domain is None:
                        # No way of inferring domain of STCs programmatically
                        print('Assuming time-domain source-space data. If data is actually freq- or time-freq-domain, specify this explicitly')
                        self.domain = 'time'
                    else:
                        self.domain = source_domain
            else:
                self.space = 'sensor'
                if 'freqs' in attrs:
                    self.datatype = 'tfr'
                    self.domain = 'time-freq'
                else:
                    self.datatype = 'epo'
                    self.domain = 'time'
        # Add important attributes
        for attr in ['times', 'freqs', 'subject', 'tstep']:
            if attr in dir(inst):
                setattr(self, attr, getattr(inst, attr))
        if self.space == 'sensor':
            self.info = inst.info #: :class:`mne.Info`: MNE Info object for data.
        elif self.space == 'source':
            self.vertices = inst.vertices #: ``list``: List of vertices copied from stc object.
        if source_freqs is not None:
            self.freqs = np.array(source_freqs)
        # Get names of data dimensions
        self.dimnames = ()
        # Start with spatial dimension
        if self.space == 'sensor':
            self.dimnames += ('chan',)
        elif self.space == 'source':
            if self.source_type == 'surface':
                self.dimnames += ('vert',)
            elif self.source_type == 'volume':
                self.dimnames += ('vox',)
        # Get names of non-spatial dimensions
        if self.domain == 'time':
            self.dimnames += ('time',)
        elif self.domain == 'freq':
            self.dimnames += ('freq',)
        elif self.domain == 'time-freq':
            self.dimnames += ('freq', 'time')
        # Get shape of data, ignoring epochs dimension if any
        if self.space == 'sensor':
            data = inst.get_data()
            if '__len__' in dir(inst):
                # Ignore epoch dimension
                self.shape = data.shape[1:]
            else:
                self.shape = data.shape
        elif self.space == 'source':
            if self.domain == 'time-freq':
                # Data will be freqs x times
                self.shape = (inst.data.shape[0],
                              len(self.freqs),
                              len(self.times))
            else:
                self.shape = inst.data.shape
        assert len(self.shape) == len(self.dimnames)
        self.size = np.prod(self.shape)
        self.ndim = len(self.dimnames) #: ``int``: Number of dimensions in data.