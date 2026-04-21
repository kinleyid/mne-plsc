
import mne
# import pyls
import pyplsc
import numpy as np
from matplotlib import pyplot as plt
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
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers, and each observation will be a participant's average in a within-participants condition.
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
        Specifies which statistic should be computed on each bootstrap iteration. The default is ``'score-covariate-corr'``.
    svd_method : ``str``, optional
        The method of SVD decomposition. The default is ``'lapack'``.
    random_state : ``int``, optional
        Random state for seeding the model. The default is None.

    Returns
    -------
    :class:`PLSC`
        PLSC object fit to the data.
    """
    
    datamat = utils.get_datamat(data)
    template = Template(data)
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
        The M/EEG data to analyze. For single-participant analysis, this should be an instance of one of MNE's data containers and each observation will be a single trial. For group-level analysis, this should be an iterable of MNE data containers, and each observation will be a participant's average in a within-participants condition.
    design : ``pd.DataFrame``, optional
        Design matrix containing indicators of experimental condition and/or covariates. The default is ``None``.
    between : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of between-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    within : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of within-participants conditions, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``.
    participant : iterable | ``str``, optional
        An iterable containing indicators (integer or string labels) of participant identity, or a string specifying which column in ``design`` contains such an indicator. The default is ``None``. This is required only if there is a within-participants condition.
    boot_stat : ``str``, optional
        Specifies which statistic should be computed on each bootstrap iteration. The default is ``'score-covariate-corr'``.
    svd_method : ``str``, optional
        The method of SVD decomposition. The default is ``'lapack'``.
    random_state : ``int``, optional
        Random state for seeding the model. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
    """
    
    datamat = utils.get_datamat(data)
    template = Template(data)
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
    within : TYPE, optional
        DESCRIPTION. The default is None.
    boot_stat : TYPE, optional
        DESCRIPTION. The default is 'score-covariate-corr'.
    svd_method : TYPE, optional
        DESCRIPTION. The default is 'lapack'.
    random_state : TYPE, optional
        DESCRIPTION. The default is None.

    Returns
    -------
    TYPE
        DESCRIPTION.
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
    PLSC model.
    """
    
    def __init__(self, template, model, grouping):
        self.template = template
        """
        :type: `int`
        A template containing xyz :class:`Template`
        """
        self.model = model
        self.grouping = grouping # Determines how certain plots will look
        self._clustering_done = False
        self.null_dist = None
    '''
    def get_labels(self, per='lv', zipped=True):
        if self.grouping_ == 'both':
            if per == 'lv':
                between = self.cond_labels_['between'].repeat(len(self.cond_labels_['within']))
                within = self.cond_labels_['within'].to_list()*len(self.cond_labels_['between'])
            elif per == 'obs':
                between_idx, within_idx, _ = self.pls.design_.T
                between = self.cond_labels_['between'][between_idx]
                within = self.cond_labels_['within'][within_idx]
            if zipped:
                labels = zip(between, within)
            else:
                labels = (between, within)
        elif self.grouping_ == 'neither':
            labels = None
        else:
            lv_labels = self.cond_labels_[self.grouping_]
            if per == 'lv':
                labels = lv_labels
            elif per == 'obs':
                level_idx = compute._get_stratifier(self.pls.design_)
                labels = lv_labels[level_idx]
                
        return labels
    '''
    '''
    def summary(self):
        n_lv = len(self.pls.singular_vals_)
        # Format for printing p values
        if self.perm_done:
            n_digs = np.ceil(np.log10(self.pls_results.inputs.n_perm)) + 2
            pval_fmt = '%%.%df' % n_digs
        print('lv_idx   var.exp.   pval')
        for lv_idx in range(n_lv):
            print("{:<9}".format(lv_idx), end='')
            # print('lv_idx %s:' % lv_idx)
            # print('%s')
            var_exp = '%s%%' % round(100*self.pls.variance_explained_[lv_idx], 2)
            print("{:<11}".format(var_exp), end='')
            # print('  %s%% var. exp.' % round(100*self.pls_results.varexp[lv_idx], 2))
            if 'pvals' in dir(self.pls):
                pval = self.pls_results.permres.pvals[lv_idx]
                print(pval_fmt % pval, end='')
                if pval < 0.001:
                    print('***')
                elif pval < 0.01:
                    print('**')
                elif pval < 0.05:
                    print('*')
                elif pval < 0.1:
                    print('.')
                else:
                    print('')
            else:
                print('(none)')
    '''
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
        p values are available through the :attr:`model.pvals_` attribute

        Examples
        --------
        >>> res.permute(n_perm=1000, n_jobs=-1)
        >>> print(res.model.pvals_)
        """
        
        #: Null distribution
        self.null_dist = self.model.permute(n_perm=n_perm,
                                            n_jobs=n_jobs,
                                            print_prog=print_prog,
                                            return_null_dist=store_null_dist)
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
        if all_channels_adjacent == 'auto':
            if self.template.datatype == 'epo':
                all_channels_adjacent = True
                print('Defaulting to all channels adjacent for ERP analysis')
            else:
                all_channels_adjacent = False
        if all_channels_adjacent:
            ch_adj = np.ones((self.template.info['nchan'],)*2)
        else:
            if montage_name is None:
                ch_adj, _ = mne.channels.find_ch_adjacency(self.template.info, 'eeg') # TODO: other options than eeg
            else:
                ch_adj, _ = mne.channels.read_ch_adjacency(montage_name)
        dim_adjs = (ch_adj,) + self.template.shape[1:]
        self.template.adjacency = mne.stats.combine_adjacency(*dim_adjs)
    def cluster(self, which='saliences', threshold=None, signed='auto'):
        """
        Identify clusters of adjacent saliences above some threshold.

        Parameters
        ----------
        which : str, optional
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be used for clustering. The default is `'saliences'`.
        threshold : float, optional
            Saliences must be above this threshold to be part of a cluster. The default is ``None``, which uses the mean salience if ``which='saliences'`` and a value of 2 if ``which='z-scores'``.
        signed : bool, optional
            If ``True``, each cluster will contain only positive or only negative saliences. If ``False``, clusters can contain both positive and negative saliences. In ERP analysis, both positive and negative saliences could be considered part of the same component. The default is ``'auto'``, which is ``False`` for ERP analysis and ``True`` for other analyses.

        Returns
        -------
        None
            None. Adds the attribute :attr:`clusters`.
        """
        _check_str_arg('which', which, ('saliences', 'z-scores'))
        if 'adjacency' not in dir(self.template):
            raise ValueError('Adjacency must be added with .add_adjacency() before clustering can be done')
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
        if callable(threshold):
            threshold = np.apply_along_axis(func1d=threshold,
                                            axis=0,
                                            arr=data)
        # In case threshold is a scalar, repeat per LV
        try:
            len(threshold)
        except:
            threshold = [threshold]*self.model.n_sv_
        
        absdata = np.abs(data)
        if signed == 'auto':
            if self.template.datatype == 'epo':
                signed = False
            else:
                signed = True
        if not signed:
            data = absdata
        
        clusters = []
        for lv_idx in range(data.shape[1]):
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
                peak_idx = clust_idx[absdata[clust_idx, lv_idx].argmax()]
                # Get coords of max
                peak_coords = np.unravel_index(peak_idx, self.template.shape)
                peaks.append(peak_coords)
            clusters.append({
                'info': {
                    'which': which,
                    'threshold': curr_thresh,
                    'n_above_thresh': n_above_thresh
                },
                'clusters': [{'idx': idx, 'peak': peak} for idx, peak in zip(idxs, peaks)]
            })
        self.clusters = clusters
        self._clustering_done = True
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
            ylabel = 'z score'
            clabel = 'Mean z score'
        elif which == 'saliences':
            data = self.model.data_sals_[:, lv_idx]
            ylabel = 'Salience'
            clabel = 'Mean salience'
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
                                 ylabel=ylabel,
                                 ythresh=ythresh)
        elif self.template.datatype == 'tfr':
            # Show average
            tf_data = data.mean(axis=0)
            viz.tfr_image(template=self.template,
                          data=tf_data,
                          clabel=clabel,
                          ax=ax)
    def plot_lv(self, lv_idx, which='saliences'):
        """
        Create a two-panel summary plot of a latent variable pair. The left panel displays the value of :attr:`boot_stat` while the right panel displays the brain saliences.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which the plot should be generated.
        which : str, optional
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be plotted in the right panel. The default is `'saliences'`.

        Returns
        -------
        f, ax
            Figure and axes containing plot.
        """
        
        f, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[2, 3])
        self.plot_boot_stat(lv_idx, ax=ax[0])
        self.plot_brain_sals(lv_idx, ax=ax[1], which=which)
        plt.tight_layout()
        return f, ax
    def plot_cluster_sizes(self, lv_idx, size_measure='pct-strong', logx=False, ax=None):
        """
        Create a plot of cluster sizes from largest to smallest.

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which cluster sizes should be plotted.
        size_measure: str, optional
            Specifies how cluster size should be measured. See :meth:`get_cluster_sizes`. The default is `'pct-strong'`.
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
        out = viz.plot_cluster_sizes(cluster_sizes=cluster_sizes,
                                     size_measure=size_measure,
                                     logx=logx,
                                     ax=ax)
        return out
    def plot_clusters(self, lv_idx, cluster_idx=None, min_size=10, size_measure='pct-strong', non_chan_plot='masked-data', separate_figures='auto'):
        """
        Plot clusters of strong loadings. 

        Parameters
        ----------
        lv_idx : int
            Index of latent variable pair for which clusters should be plotted.
        cluster_idx : indexer, optional
            Index of cluster(s) to display. The default is ``None``, which displays all clusters whose size exceeds ``min_size``
        min_size : TYPE, optional
            Minimum size of clusters to display. The default is ``10``. Ignored if ``cluster_idx`` is specified.
        size_measure : str, optional
            Specifies the size measure to use when comparing cluster sizes to ``min_size``. See :meth:`get_cluster_sizes`. The default is ``'pct-strong'``. Ignored if ``cluster_idx`` is specified.
        non_chan_plot : TYPE, optional
            DESCRIPTION. The default is ``'masked-data'``.
        separate_figures : bool, optional
            Specifies whether each cluster should be displayed in a separate figure. The default is ``'auto'``, which displays clusters in separate figures if there are more than 4.

        Returns
        -------
        None
        """
        
        lv_clusters = self.clusters[lv_idx]
        if lv_clusters['info']['which'] == 'saliences':
            data = self.model.data_sals_[:, lv_idx]
        elif lv_clusters['info']['which'] == 'z-scores':
            data = self.model.data_sals_z_[:, lv_idx]
        # Default to manually specified cluster_idx
        if cluster_idx is None:
            # Plot all clusters above the min size
            cluster_sizes = self.get_cluster_sizes(lv_idx=lv_idx,
                                                   size_measure=size_measure)
            cluster_idx = np.where(cluster_sizes >= min_size)[0]
        # Fallback to checking cluster sizes
        try:
            len(cluster_idx)
        except:
            # Presumably an integer
            cluster_idx = [cluster_idx]
        if len(cluster_idx) == 0:
            raise ValueError('No clusters meet or exceed the minimum cluster size')
        if separate_figures == 'auto':
            separate_figures = len(cluster_idx) > 4
        if not separate_figures:
            f, ax = plt.subplots(nrows=len(cluster_idx))
            if len(cluster_idx) == 1:
                ax = [ax] # Make subscriptable for later on
        for i in cluster_idx:
            if separate_figures:
                f, curr_ax = plt.subplots()
            else:
                curr_ax = ax[i]
            viz.plot_cluster(data=data,
                             template=self.template,
                             cluster=lv_clusters['clusters'][i],
                             cluster_info=lv_clusters['info'],
                             non_chan_plot=non_chan_plot,
                             ax=curr_ax)

class MCPLSC(PLSC):
    """
    Mean-centred PLSC
    """
    
    def get_marginal_brain_scores(self, lv_idx, margin):
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
                allowed_datatypes = ('epo', 'tfr')
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
        groupwise_means = pyplsc.utils.get_groupwise_means(
            data=self.model.data_,
            group_idx=self.model.stratifier_)
        hadamards = groupwise_means * loadings
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
        
        scores = self.get_marginal_brain_scores(lv_idx=lv_idx, margin=margin)
        labels = self.model.design_sal_labels_
        out = viz.plot_marginal_brain_scores(scores=scores,
                                             margin=margin,
                                             labels=labels,
                                             template=self.template,
                                             grouping=self.grouping)
        return out

class Template():
    """
    Template for plotting
    """
    def __init__(self, source):
        # Keep the useful info without the data
        if isinstance(source, list):
            source = source[0]
        self.datatype = utils.infer_datatype(source)
        if utils.is_epochs(source, datatype=self.datatype):
            self.shape = source._data.shape[1:]
        else:
            self.shape = source._data.shape
        self.size = np.prod(self.shape)
        dimnames = {
            'epo': ('chan', 'time'),
            'spec': ('chan', 'freq'),
            'tfr': ('chan', 'freq', 'time')}
        self.dimnames = dimnames[self.datatype]
        self.info = source.info
        for attr in ['times', 'freqs']:
            if attr in dir(source):
                setattr(self, attr, getattr(source, attr))
