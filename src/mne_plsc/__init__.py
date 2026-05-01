
import mne
import pyplsc
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import gridspec
from mne.stats.cluster_level import _find_clusters

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
    
    template = Template(data)
    datamat = utils.get_datamat(data, datatype=template.datatype)
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
           effects='all',
           boot_stat='condwise-scores-centred',
           svd_method='lapack',
           random_state=None):
    """
    Fit mean-centred PLS model.

    Parameters
    ----------
    data : MNE object or iterable of MNE objects
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers for epoched data (e.g., :class:`mne.Epochs`) and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers for averages over epochs (e.g., :class:`mne.Evoked`), and each observation will be a participant's average in a within-participants condition.
    design : ``pd.DataFrame``, optional
        Design matrix containing indicators of experimental condition and/or covariates. The default is ``None``.
    between : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of between-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    within : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of within-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    participant : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of participant identity, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``. This is required only if there is a within-participants condition.
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
    
    template = Template(data)
    datamat = utils.get_datamat(data, datatype=template.datatype)
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
    datamat_list = [utils.get_datamat(ptpt) for ptpt in data]
    design_list = [ptpt.metadata for ptpt in data]
    template = Template(data)
    model = pyplsc.WPLSC(boot_stat=boot_stat,
                         svd_method=svd_method,
                         random_state=random_state)
    model.fit(data=datamat_list,
              design=design_list,
              covariates=covariates,
              within=within)
    return PLSC(template, model, grouping='within')

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
    def add_source_info(self, src=None, mri=None, subjects_dir=None):
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
            self.template.src = src
        if mri is not None:
            self.template.mri = mri
        if subjects_dir is not None:
            self.template.subjects_dir = subjects_dir
    def add_adjacency(self, all_channels_adjacent='auto', montage_name=None):
        """
        Add adjacency matrix for clustering.

        Parameters
        ----------
        all_channels_adjacent : bool, optional
            Specifies whether all channels should be considered adjacent to each other for the purposes of clustering. This is useful when doing ERP analyses, where strong loadings at non-adjacent channels would be considered part of the same component. The default is ``'auto'``, which is ``True`` for ERP analysis (inferred based on :attr:`template.datatype`) and ``False`` for all other analyses.
        montage_name : str, optional
            Name of montage passed to ``mne.channels.read_ch_adjacency``. The default is ``None``, which uses ``mne.channels.find_ch_adjacency`` to get channel adjacency.

        Returns
        -------
        None.
            None. Adds an ``adjacency`` attribute to :attr:`template` which indicates which channels, times, and frequencies (as applicable) are adjacent for clustering.
        """
        if self.template.space == 'sensor':
            if all_channels_adjacent == 'auto':
                if self.template.datatype == 'epo':
                    all_channels_adjacent = True
                    print('Defaulting to all channels adjacent for ERP/ERF analysis')
                else:
                    all_channels_adjacent = False
            if all_channels_adjacent:
                ch_adj = np.ones((self.template.info['nchan'],)*2)
            else:
                if montage_name is None:
                    ch_types = set(self.template.info.get_channel_types())
                    if len(ch_types) > 1:
                        raise ValueError('Multiple channel types present in data: %s. Adjacency could not be computed' % ch_types)
                    ch_type = ch_types.pop() # One-element set
                    ch_adj, _ = mne.channels.find_ch_adjacency(self.template.info, ch_type)
                else:
                    ch_adj, _ = mne.channels.read_ch_adjacency(montage_name)
            spatial_adj = ch_adj
        elif self.template.space == 'source':
            if self.template.src is None:
                raise ValueError('Source space must be specified to compute spatial adjacency. See add_src()')
            # TODO: other options for surface source spaces
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
        # Auto-determine data to cluster
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
                # Conventional 2 BSR
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
            if self.template.datatype in ['epo', 'surf-stc', 'vol-stc']:
                print('Defaulting to unsigned clustering')
                signed = False
            else:
                signed = True
        if not signed:
            data = absdata
        
        clusters = []
        for lv_idx in range(self.model.rank_):
            # TODO: make an option for separate negative + positive clusters
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
    def _get_cluster(self, lv_idx, cluster_idx, return_data=True):
        # TODO: give this a more obscure name because it's obscure
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
            # Get data used for clustering, reshaped
            if info['which'] == 'saliences':
                data = self.model.data_sals_[:, lv_idx]
            elif info['which'] == 'z-scores':
                data = self.model.data_sals_z_[:, lv_idx]
            reshaped = data.copy().reshape(self.template.shape)
            out += (reshaped,)
        return out
    def cluster_to_stc(self, lv_idx, cluster_idx):
        cluster, _, data = self._get_cluster(lv_idx, cluster_idx)
        data[~cluster['mask']] = 0
        times = self.template.times * 1000 # s to ms
        tstep = np.diff(times)[0]
        stc = mne.SourceEstimate(data=data,
                                 vertices=self.template.vertices,
                                 tmin=times[0],
                                 tstep=tstep,
                                 subject=self.template.subject) # TODO: does this make sense in general?
        return stc
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
        SUMMARY.

        Parameters
        ----------
        lv_idx : TYPE, optional
            DESCRIPTION. The default is None.
        cluster_idx : TYPE, optional
            DESCRIPTION. The default is None.

        Returns
        -------
        TYPE
            DESCRIPTION.
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
    def plot_boot_stat(self, lv_idx, ax=None):
        """
        Visualize :attr:`model.boot_stat` with a barplot.

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
        df = self.model.get_boot_stat_frame(lv_idx)
        out = viz.boot_stat_barplot(df=df,
                                    boot_stat=self.model.boot_stat,
                                    grouping=self.grouping,
                                    with_ci=self.model._boot_done,
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
        if self.template.datatype in ['epo', 'spec']:
            # Line plot with spatial colours
            if self.template.datatype == 'epo':
                x = self.template.times
                xlabel = 'Time (s)'
            elif self.template.datatype == 'spec':
                x = self.template.freqs
                xlabel = 'Frequecy (Hz)'
            if self._clustering_done:
                ythresh = self.clusters[lv_idx]['info']['threshold']
            else:
                ythresh = None
            viz.channel_lineplot(x=x,
                                 ch_y=data,
                                 info=self.template.info,
                                 ax=ax,
                                 xlabel=xlabel,
                                 ylabel=label,
                                 ythresh=ythresh)
        elif self.template.datatype == 'tfr':
            # Show average
            tf_data = data.mean(axis=0)
            viz.plot_labeled_raster(template=self.template,
                                    data=tf_data,
                                    xdim='time',
                                    ydim='freq',
                                    vlabel=avg_label,
                                    ax=ax)
        elif self.template.datatype in ['surf-stc', 'vol-stc']:
            viz.plot_labeled_raster(template=self.template,
                                    data=tf_data,
                                    xdim='time',
                                    ydim='vert',
                                    vlabel=label,
                                    ax=ax)
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
    def plot_cluster_sizes(self, lv_idx, size_measure='pct-strong', n_clust=None, logx=False, ax=None):
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
        logx : bool, optional
            If ``True``, x axis will be on a log scale. Default is ``False``.
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
                                     logx=logx,
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
            elif self.template.datatype in ['tfr']:
                plot_type = 'raster'
            elif self.template.datatype in ['surf-stc', 'vol-stc']:
                plot_type = 'distribution'
        cluster, info, data = self._get_cluster(lv_idx, cluster_idx)
        # TODO: fail gracefully in case of mismatch between datatype and plot_type
        if plot_type == 'butterfly':
            out = viz.plot_cluster_butterfly(data=data,
                                             template=self.template,
                                             cluster=cluster,
                                             which=info['which'],
                                             ythresh=info['threshold'],
                                             highlight=highlight,
                                             ax=ax)
        elif plot_type == 'raster':
            out = viz.plot_cluster_raster(data=data,
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
        
        if self.template.datatype == 'surf-stc':
            # Interactive surface plot
            # Determine whether cluster peak is in left or right hemisphere
            cluster, cluster_info = self._get_cluster(lv_idx, cluster_idx, return_data=False)
            vert_peak, time_peak = cluster['peak_coords']
            if vert_peak < self.template.vertices[0].size:
                hemi = 'lh'
            else:
                hemi = 'rh'
            # Convert to STC object for plotting
            stc = self.cluster_to_stc(lv_idx, cluster_idx)
            # Set colour limits
            cmax = np.abs(stc.data.flat[cluster['peak_flat']])
            cmin = cluster_info['threshold']
            cmid = cmin
            clim = (cmin, cmid, cmax)
            # Generate interactive plot
            out = stc.plot(subjects_dir=self.template.subjects_dir,
                           hemi=hemi,
                           time_viewer=False,
                           initial_time=stc.times[time_peak],
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
        _check_str_arg('highlight', highlight, ['peak', 'extent']) # Can't be none here, even though it can be non for the non-spatial plot
        # Default to manually specified cluster_idx
        f, curr_ax = plt.subplots(layout='constrained')
        if self.template.datatype == 'surf-stc':
            raise ValueError('Spatial and non-spatial cluster visualizations cannot be shown in the same figure. Call .plot_cluster_nonspatial() and .plot_cluster_spatial() separately.')
        else:
            # Subdivide into left and right axis
            sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=curr_ax.get_subplotspec())
            curr_ax.remove()
            # Left axis: visualize non-spatial dimension(s)
            ax_left = f.add_subplot(sub_gs[0])
            self.plot_cluster_nonspatial(lv_idx=lv_idx,
                                         cluster_idx=cluster_idx,
                                         plot_type=plot_type,
                                         highlight=highlight,
                                         ax=ax_left)
            ax_right = f.add_subplot(sub_gs[1])
            self.plot_cluster_spatial(lv_idx=lv_idx,
                                      cluster_idx=cluster_idx,
                                      highlight=highlight,
                                      ax=ax_right)

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
        - ``'neither'``
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
    def __init__(self, source):
        # Document attributes
        self.src = None #: :class:`mne.SourceSpaces`: Source spaces of data, if applicable.
        self.mri = None #: Niimg-like: Structural MRI data, if applicable.
        self.subjects_dir = None #: path-like: Freesurfer subjects directory, if applicable.
        # Keep the useful info without the data
        if isinstance(source, list):
            source = source[0]
        # Infer datatype
        self.datatype = utils.infer_datatype(source) #: ``str``: Specifies the type of the data.
        # Determine sensors space vs source space
        if self.datatype in ['epo', 'spec', 'tfr']:
            space = 'sensor'
        elif self.datatype in ['surf-stc', 'vol-stc']:
            space = 'source'
        self.space = space #: ``str``: Specifies whether the data is in sensor or source space
        # Get shape of data, ignoring epochs dimension
        if self.datatype in ['surf-stc', 'vol-stc']:
            data = source.data
        else:
            data = source.get_data()
        if utils.is_epochs(source, datatype=self.datatype):
            shape = data.shape[1:]
        else:
            shape = data.shape
        self.shape = shape #: ``tuple``: Specifies the original shape of the data.
        self.size = np.prod(self.shape) #: ``int``: Size of data.
        # Get names of data dimensions
        dimnames = {
            'epo':      ('chan', 'time'),
            'spec':     ('chan', 'freq'),
            'tfr':      ('chan', 'freq', 'time'),
            'vol-stc':  ('vert', 'time'),
            'surf-stc': ('vert', 'time')}
        self.dimnames = dimnames[self.datatype] #: ``tuple``: Names of dimensions of data.
        self.ndim = len(self.dimnames) #: ``int``: Number of dimensions in data.
        if self.space == 'sensor':
            self.info = source.info #: :class:`mne.Info`: MNE Info object for data.
        elif self.space == 'source':
            self.vertices = source.vertices #: ``list``: List of vertices copied from stc object.
        self.times = None #: ``numpy.ndarray``: Times, copied from data
        self.freqs = None #: ``numpy.ndarray``: Frequencies, copied from data
        self.subject = None #: ``str``: Freesurfer subject name, copied from data
        for attr in ['times', 'freqs', 'subject']:
            if attr in dir(source):
                setattr(self, attr, getattr(source, attr))
