
import mne
# import pyls
import pyplsc
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd
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

def fit_beh_pls(data,
                covariates,
                design=None,
                between=None,
                within=None,
                participant=None,
                boot_stat='score-covariate-corr',
                svd_method='lapack',
                random_state=None):
    datamat = utils.get_datamat(data)
    template = Template(data)
    model = pyplsc.PLSC(boot_stat,
                        svd_method,
                        random_state)
    model.fit(data=datamat,
              design=design,
              between=between,
              within=within,
              participant=participant)
    grouping = utils.get_grouping(between, within)
    return PLS(template, model, grouping)

def fit_mc_pls(data,
               design=None,
               between=None,
               within=None,
               participant=None,
               effects='all',
               boot_stat='condwise-scores-centred',
               svd_method='lapack',
               random_state=None):
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
    return MCPLS(template, model, grouping)

class PLS():
    def __init__(self, template, model, grouping):
        self.template = template
        self.model = model
        self.grouping = grouping # Determines how certain plots will look
        self._clustering_done = False
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
    def add_adjacency(self, all_channels_adjacent='auto', montage_name=None):
        """
        Add adjacency matrix for clustering.

        Parameters
        ----------
        all_channels_adjacent : bool, optional
            Specifies whether all channels should be considered adjacent to each other for the purposes of clustering. This is useful when doing ERP analyses, where strong loadings at non-adjacent channels would be considered part of the same component. The default is ``'auto'``, which is ``True`` for ERP analysis (inferred based on :attr:`template.datatype`) and ``False`` for all other analyses.
        montage_name : str, optional
            Name of montage passed to ``mne.channels.read_ch_adjacency``. The default is None, which uses ``mne.channels.find_ch_adjacency`` to get channel adjacency.

        Returns
        -------
        None. Adds an ``adjacency`` property to :attr:`template` which indicates which channels, times, and frequencies (as applicable) are adjacent for clustering.
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
            Specifies whether raw saliences (``'saliences'``) or z scores (``'z-scores'``) should be used for clustering. The default is 'saliences'.
        threshold : float, optional
            Saliences must be above this threshold to be part of a cluster. The default is ``None``, which uses the mean salience if ``which='saliences'`` and a value of 2 if ``which='z-scores'``.
        signed : bool, optional
            If ``True``, each cluster will contain only positive or only negative saliences. If ``False``, clusters can contain both positive and negative saliences. In ERP analysis, both positive and negative saliences could be considered part of the same component. The default is ``'auto'``, which is ``False`` for ERP analysis and ``True`` for other analyses.

        Returns
        -------
        None
        """
        _check_str_arg('which', which, ('saliences', 'z-scores'))
        if 'adjacency' not in dir(self.template):
            raise ValueError('Adjacency must be added with .add_adjacency() before clustering can be done')
        if which == 'z-scores':
            if not self.model._boot_done:
                raise ValueError('Bootstrap resampling must be done to use z scores for clustering.')
            data = self.model.bootstrap_ratios_
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
            threshold = [threshold]*self.model.n_lv_
        
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
            For which latent variable index should cluster sizes be returned? The default is 0 (first latent variable).
        size_measure : str, optional
            How should cluster size be measured? Must be one of:
            
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
    def plot_scree(self, perm_dist=None):
        # TODO: implement
        raise NotImplementedError()
    def plot_scores(self, lv_idx, ax=None):
        """
        Create a scatterplot of data scores against design scores.

        Parameters
        ----------
        lv_idx : int
            Index of the latent variable pair to plot.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is None, which generates a new figure.

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
            Index of latent variable for which the plot should be generated.
        ax : instance of Matplotlib Axes, optional
            Axes to plot to. The default is None, which generates a new figure.

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
        if which == 'z-scores':
            data = self.model.bootstrap_ratios_[:, lv_idx]
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
    def plot_lv(self, lv_idx=0, which='saliences', show=True):
        if which == 'z-scores':
            assert self.boot_done
        f, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[2, 3])
        self.plot_boot_stat(lv_idx, ax=ax[0])
        self.plot_brain_sals(lv_idx, ax=ax[1], which=which)
        plt.tight_layout()
    def plot_cluster_sizes(self, lv_idx=0, size_measure='pct-strong', logx=False, ax=None):
        cluster_sizes = self.get_cluster_sizes(lv_idx=lv_idx,
                                               size_measure=size_measure)
        out = viz.plot_cluster_sizes(cluster_sizes=cluster_sizes,
                                     size_measure=size_measure,
                                     logx=logx,
                                     ax=ax)
        return out
    def plot_clusters(self, lv_idx=0, cluster_idx=None, min_size=10, size_measure='pct-strong', non_chan_plot='masked-data', separate_figures='auto'):
        lv_clusters = self.clusters[lv_idx]
        if lv_clusters['info']['which'] == 'saliences':
            data = self.model.data_sals_[:, lv_idx]
        elif lv_clusters['info']['which'] == 'z-scores':
            data = self.model.bootstrap_ratios_[:, lv_idx]
        if cluster_idx is None:
            # Plot all clusters above the min size
            cluster_sizes = self.get_cluster_sizes(lv_idx=lv_idx,
                                                   size_measure=size_measure)
            cluster_idx = np.where(cluster_sizes >= min_size)[0]
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

class MCPLS(PLS):
    def get_marginal_brain_scores(self, lv_idx, margin):
        _check_str_arg('margin', margin,
                       ('chan', 'time', 'freq', 'time-freq'))
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
        scores = self.get_marginal_brain_scores(lv_idx=lv_idx, margin=margin)
        labels = self.model.design_sal_labels_
        out = viz.plot_marginal_brain_scores(scores=scores,
                                             margin=margin,
                                             labels=labels,
                                             template=self.template,
                                             grouping=self.grouping)
        return out

class Template():
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
