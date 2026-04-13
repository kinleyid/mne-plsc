
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
                # TODO: validate that channel locations are added
                ch_adj, _ = mne.channels.find_ch_adjacency(self.template.info, 'eeg')
            else:
                ch_adj, _ = mne.channels.read_ch_adjacency(montage_name)
        dim_adjs = (ch_adj,) + self.template.shape[1:]
        self.template.adjacency = mne.stats.combine_adjacency(*dim_adjs)
    def cluster(self, which='saliences', threshold=None, signed='auto'):
        _check_str_arg('which', which, ('saliences', 'bootstrap-ratios'))
        if 'adjacency' not in dir(self.template):
            raise ValueError('Adjacency must be added with .add_adjacency() before clustering can be done')
        if which == 'bootstrap-ratios':
            if not self.model._boot_done:
                raise ValueError('Bootstrap resampling must be done to use bootstrap ratios for clustering.')
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
    def get_cluster_sizes(self, lv_idx=0, size_measure='pct-strong'):
        cluster_set = self.clusters[lv_idx]
        abs_sizes = np.array([len(c['idx']) for c in cluster_set['clusters']])
        if size_measure == 'absolute':
            sizes = abs_sizes
        elif size_measure == 'pct-strong':
            sizes = 100*abs_sizes/cluster_set['info']['n_above_thresh']
        elif size_measure == 'pct-total':
            sizes = 100*abs_sizes/self.template.size
        return sizes
    def plot_scree(self):
        # TODO: implement
        raise NotImplementedError()
    def plot_scores(self, lv_idx=0, ax=None):
        df = self.model.get_scores_frame(lv_idx)
        out = viz.score_scatterplot(df, self.grouping, ax=ax)
        return out
    def plot_boot_stat(self, lv_idx, ax=None):
        df = self.model.get_boot_stat_frame(lv_idx)
        out = viz.boot_stat_barplot(df=df,
                                    boot_stat=self.model.boot_stat,
                                    grouping=self.grouping,
                                    with_ci=self.model._boot_done,
                                    ax=ax)
        return out
    def plot_brain_weights(self, lv_idx, which='saliences', ax=None):
        if which == 'bootstrap-ratios':
            data = self.model.bootstrap_ratios_[:, lv_idx]
            ylabel = 'Bootstrap ratio'
            clabel = 'Mean boostrap ratio'
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
        if which == 'bootstrap-ratios':
            assert self.boot_done
        f, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[2, 3])
        self.plot_boot_stat(lv_idx, ax=ax[0])
        self.plot_brain_weights(lv_idx, ax=ax[1], which=which)
        plt.tight_layout()
    def plot_cluster_sizes(self, lv_idx=0, size_measure='pct-strong', logx=False, ax=None):
        _check_str_arg('size_measure', size_measure,
                       ('pct-strong', 'pct-total', 'absolute'))
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
        elif lv_clusters['info']['which'] == 'bootstrap-ratios':
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

class Results():
    def __init__(self, template, datatype, submatrices, labels, pls_type, pls_results, cov_labels=None):
        self.template = template
        self.shape = template._data.shape
        self.datatype = datatype
        self.submatrices = submatrices
        self.labels = labels
        self.pls_type = pls_type
        self.pls_results = pls_results
        self.cov_labels = cov_labels
    def summary(self):
        # Display results
        n_lv = len(self.pls_results.singvals)
        # Format for printing p values
        n_digs = np.ceil(np.log10(self.pls_results.inputs.n_perm)) + 2
        pval_fmt = '%%.%df' % n_digs
        print('lv_idx   var.exp.   pval')
        for lv_idx in range(n_lv):
            print("{:<9}".format(lv_idx), end='')
            # print('lv_idx %s:' % lv_idx)
            # print('%s')
            var_exp = '%s%%' % round(100*self.pls_results.varexp[lv_idx], 2)
            print("{:<11}".format(var_exp), end='')
            # print('  %s%% var. exp.' % round(100*self.pls_results.varexp[lv_idx], 2))
            if 'pvals' in dir(self.pls_results.permres):
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
                print('N/A')
    def flip_signs(self, lv_idx=None):
        if lv_idx is None:
            lv_idx = list(range(len(self.pls_res.singvals)))
        self.pls_results.x_weights[:, lv_idx] *= -1
        self.pls_results.y_weights[:, lv_idx] *= -1
        self.pls_results.bootres.x_weights_normed[:, lv_idx] *= -1
        ## TODO: check how to index LVs
        self.pls_results.bootres.contrast *= -1
    
    def get_template(self, lv_idx=0, which='ratios'):
        template = self.template.copy()
        if which == 'bootstrap-ratios':
            data = self.pls_results.bootres.x_weights_normed[:, lv_idx]
        elif which == 'saliences':
            data = self.pls_results.x_weights[:, lv_idx]
        data = data.reshape(template._data.shape)
        template._data = data
        return template
    
    def add_adjacency(self, all_channels_adjacent=False, montage_name=None):
        if all_channels_adjacent:
            ch_names = self.template.info['ch_names']
            ch_adj = np.ones((len(ch_names),)*2)
        else:
            if montage_name:
                ch_adj, ch_names = mne.channels.read_ch_adjacency(montage_name)
            else:
                # TODO: validate that channel locations are added
                ch_adj, ch_names = mne.channels.find_ch_adjacency(self.template.info, 'eeg')
        if self.datatype == 'epochs':
            # data shape is channels x time
            dim_adjs = (ch_adj, len(self.template.times))
        elif self.datatype == 'spec':
            dim_adjs = (ch_adj, len(self.template.freqs))
        elif self.datatype == 'tfr':
            dim_adjs = (ch_adj, len(self.template.freqs), len(self.template.times))
        self.adjacency = mne.stats.combine_adjacency(*dim_adjs)

    def cluster(self, what='bootstrap-ratios', threshold=2, signed=True):
        assert threshold > 0
        # TODO: ensure bootstrapping and clustering have been done
        if what == 'bootstrap-ratios':
            assert 'bootres' in dir(self.pls_results)
            data = self.pls_results.bootres.x_weights_normed
        elif what == 'saliences':
            data = self.pls_results.x_weights
        else:
            pass # TODO: error msg here
        
        if not signed:
            data = np.abs(data)
        
        clusters = []
        for lv_idx in range(data.shape[1]):
            # TODO: make an option for separate negative + positive clusters
            # Separate clustering for positive and negative
            print('Computing clusters for lv_idx %s...' % lv_idx)
            """
            if signed:
                # Compute positive and negative clusters separately
                pos_clusts, sums = mne.stats.cluster_level._find_clusters(
                    boot_rats[:, lv_idx], tail=1, threshold=threshold, adjacency=self.adjacency)
                neg_clusts, sums = mne.stats.cluster_level._find_clusters(
                    boot_rats[:, lv_idx], tail=-1, threshold=-threshold, adjacency=self.adjacency)
                clusts = pos_clusts + neg_clusts
            else:
                clusts, sums = mne.stats.cluster_level._find_clusters(
                    data[:, lv_idx], tail=0, threshold=threshold, adjacency=self.adjacency)
            """
            clusts, sums = mne.stats.cluster_level._find_clusters(
                data[:, lv_idx], tail=0, threshold=threshold, adjacency=self.adjacency)
            # Sort largest to smallest
            clusts.sort(key=len, reverse=True)
            print('%s clusters' % len(clusts))
            clusters.append(clusts)
        self.clusters = clusters
    
    def get_cluster_peaks(self, lv_idx=0, measure='bootstrap-ratios', ):
        clusters = self.clusters[lv_idx]
        if measure == 'bootstrap-ratios':
            assert 'bootres' in dir(self.pls_results)
            data = self.pls_results.bootres.x_weights_normed
        elif measure == 'saliences':
            data = self.pls_results.x_weights
        else:
            pass # TODO: error msg here
        data = np.abs(data)
        
        peaks = []
        for cluster in clusters:
            # Get linear index of max
            peak_idx = cluster[data[cluster].argmax()]
            # Get coords of max
            peak_coords = np.unravel_index(peak_idx, self.shape)
            peaks.append(peak_coords)
        return peaks
    
    def plot_cluster_sizes(self, lv_idx=0, measure='percent', logx=False):
        # Log x scale
        # TODO: Checking for percent vs absolute
        sizes = [len(c) for c in self.clusters[lv_idx]]
        if measure == 'percent':
            sizes = [100*s/self.template.data.size for s in sizes]
        idx = np.arange(len(self.clusters[lv_idx])) + 1
        f, ax = plt.subplots()
        ax.plot(idx, sizes)
        ax.set_xlabel('Cluster number')
        if measure == 'percent':
            ax.set_ylabel('Cluster size (%)')
        elif measure == 'absolute':
            ax.set_ylabel('Cluster size')
        ax.set_title('LV idx: %s' % lv_idx)
        if logx:
            ax.set_xscale('log')
        return ax
    
    def trim_clusters(self, min_size=0.05):
        data_size = self.template.data.size
        for lv_idx in range(self.pls_results.x_weights.shape[1]):
            print('lv_idx %s: ' % lv_idx, end='')
            clusts = self.clusters[lv_idx]
            print('%s -> ' % len(clusts), end='')
            if min_size < 1:
                # Relative size
                clusts = [c for c in clusts if len(c)/data_size >= min_size]
            else:
                # Absolute size
                clusts = [c for c in clusts if len(c) >= min_size]
            print('%s clusters' % len(clusts))
            self.clusters[lv_idx] = clusts
            
    def plot_xy_scores(self, lv_idx=0, axes=None):
        brain_scores = self.pls_results.x_scores[:, lv_idx]
        design_scores = self.pls_results.y_scores[:, lv_idx]
        if self.pls_type == 'mc':
            _, indic = np.where(self.pls_results.inputs.Y)
        else:
            indic = None
        if axes is None:
            f, axes = plt.subplots()
        scatter = axes.scatter(design_scores, brain_scores, c=indic)
        handles, _ = scatter.legend_elements()
        axes.set_xlabel('Design score')
        axes.set_ylabel('Brain score')
        axes.legend(handles, self.labels)
        return axes
    
    def plot_latent_var(self, lv_idx=0, which='bootstrap-ratios', show=True):
        if which == 'bootstrap-ratios':
            assert 'bootres' in dir(self.pls_results)
        f, ax = plt.subplots(nrows=1, ncols=2, width_ratios=[2, 3])
        # Get data to plot
        if which == 'bootstrap-ratios':
            # Right panel is bootstrap ratios
            right_data = self.pls_results.bootres.x_weights_normed[:, lv_idx]
            right_ylabel = 'Bootstrap ratio'
            # Left panel depends on pls type
            if self.pls_type == 'mc':
                left_data = self.pls_results.bootres.contrast[:, lv_idx]
                ci = self.pls_results.bootres.contrast_ci[:, lv_idx, :]
                yerr = ci - np.reshape(left_data, (len(left_data), 1))
                yerr = np.abs(yerr.T)
                left_ylabel = 'Brain score'
                left_labels = self.labels
            elif self.pls_type == 'beh':
                left_data = self.pls_results.bootres.y_loadings[:, lv_idx]
                ci = self.pls_results.bootres.y_loadings_ci[:, lv_idx, :]
                yerr = ci - np.reshape(left_data, (len(left_data), 1))
                yerr = np.abs(yerr.T)
                left_ylabel = 'Loading'
                left_labels = ['%s %s' % (cov_label, group_label) for cov_label in self.cov_labels for group_label in self.labels]
        elif which in ['loadings', 'saliences']:
            left_data = self.pls_results.y_weights[:, lv_idx]
            yerr = None
            left_ylabel = 'Design %s' % which[:-1]
            right_data = self.pls_results.x_weights[:, lv_idx]
            right_ylabel = 'Brain %s' % which[:-1]
            if self.pls_type == 'mc':
                left_labels = self.labels
            elif self.pls_type == 'beh':
                left_labels = self.cov_labels
        # Left axis: bar plot
        ax[0].bar(left_labels,
                  left_data,
                  yerr=yerr,
                  facecolor='gray', edgecolor='black')
        ax[0].tick_params('x', rotation=90)
        ax[0].set_ylabel(left_ylabel)
        # Right axis: depends on data type
        if self.datatype == 'epochs':
            viz.plot_lv_epochs(self.template, right_data, ax[1])
        elif self.datatype == 'spec':
            viz.plot_lv_spec(self.template, right_data, ax[1])
        elif self.datatype == 'tfr':
            viz.plot_lv_tfr(self.template, right_data, ax[1])
        ax[1].set_ylabel(right_ylabel)
        # Add title
        title_txt = 'lv_idx %s; %.2f%% var. exp.' % (lv_idx, 100*self.pls_results.varexp[lv_idx])
        if len(self.pls_results.permres) > 0:
            # Add p value
            pval = self.pls_results.permres.pvals[lv_idx]
            title_txt = '%s; p = %.4f' % (title_txt, pval)
        f.suptitle(title_txt)
        plt.tight_layout()
        if show:
            plt.show()
        return f
    
    def plot_clusters(self, lv_idx, mask_params=None):
        # TODO: separate_plots=False
        
        # Setup data
        clusts = self.clusters[lv_idx]
        n_clusts = len(clusts)
        boot_rats = self.pls_results.bootres.x_weights_normed[:, lv_idx]
        boot_rats = np.reshape(boot_rats, self.template._data.shape)
        
        # Default mask params
        if mask_params is None:
            mask_params = dict(marker='o', markerfacecolor='k', markeredgecolor='w',
                 linewidth=0, markersize=2)
        
        # Setup figure
        fig = plt.figure()
        gs = gridspec.GridSpec(
            n_clusts*2, 2,
            width_ratios=[10, 6],
            height_ratios=[1, 20]*n_clusts)
        clust_idxs = range(len(clusts))
        for clust_idx in clust_idxs:
            clust = clusts[clust_idx]
            mask = np.zeros(self.template._data.shape, bool)
            mask.flat[clust] = True
            # Add title
            title_ax = fig.add_subplot(gs[clust_idx*2, :])
            title_text = 'lv_idx %s, cluster %s' % (lv_idx, clust_idx + 1)
            # title_ax.set_title() # , fontsize=14)
            title_ax.text(
                0.5, 0,
                title_text,
                ha="center", va="center",
                fontsize=10
            )
            title_ax.axis("off")
            
            # Plot number of channels in cluster
            nchan_ax = fig.add_subplot(gs[clust_idx*2 + 1, 0])
            if self.datatype == 'epochs':
                viz.plot_clust_nchan_epochs(self.template, mask, nchan_ax)
            elif self.datatype == 'spec':
                viz.plot_clust_nchan_spec(self.template, mask, nchan_ax)
            elif self.datatype == 'tfr':
                self.template._data = np.stack([mask.sum(axis=0)]*len(self.template.info['ch_names']))
                self.template.plot(combine='mean', axes=nchan_ax, show=False)
            
            # Plot topomap
            topo_ax = fig.add_subplot(gs[clust_idx*2 + 1, 1])
            if self.datatype == 'epochs':
                time_mask = mask.sum(axis=0) > 0
                topo_data = boot_rats[:, time_mask].mean(axis=1)
                ch_mask = mask.sum(axis=1) > 0
            elif self.datatype == 'spec':
                freq_mask = mask.sum(axis=0) > 0
                topo_data = boot_rats[:, freq_mask].mean(axis=1)
                ch_mask = mask.sum(axis=1) > 0
            elif self.datatype == 'tfr':
                tf_mask = mask.sum(axis=0) > 0
                topo_data = boot_rats[:, tf_mask].mean(axis=1)
                ch_mask = mask.sum(axis=(1,2)) > 0
            
            # viz.plot_topomap(self.template, topo_data, ch_mask, topo_ax)
            mne.viz.plot_topomap(
                data=topo_data, pos=self.template.info,
                axes=topo_ax, mask=ch_mask,
                sensors=False, mask_params=mask_params,
                show=False)
                
            """
            im, _ = mne.viz.plot_topomap(masked_ch_data, template.info, axes=ax, show=False, mask=chan_mask)
            cbar = ax.figure.colorbar(im, shrink=0.6)
            cbar.ax.set_ylabel('Mean bootstrap ratio in cluster')
            """
        # plt.tight_layout()
        return fig
    
    def get_marginal_brain_scores(self, lv_idx, margin):
        # TODO: more informative error msgs here
        if margin == 'time':
            assert self.datatype in ['epochs', 'tfr']
        elif margin == 'frequency':
            assert self.datatype in ['spec', 'tfr']
        elif margin == 'time-frequency':
            assert self.datatype == 'tfr'
        
        # Compute hadamard products
        loadings = self.pls_results.x_weights[:, lv_idx]
        hadamards = []
        for idx in range(len(self.labels)):
            submatrix = self.submatrices[idx]
            hadamard = submatrix.mean(axis=0) * loadings
            hadamard = np.reshape(hadamard, self.template._data.shape)
            hadamards.append(hadamard)
        
        if margin == 'time':
            if self.datatype == 'epochs':
                non_margin_axes = 0
            elif self.datatype == 'tfr':
                non_margin_axes = (0, 1)
        elif margin == 'frequency':
            if self.datatype == 'spec':
                non_margin_axes = 0
            elif self.datatype == 'tfr':
                non_margin_axes = (0, 2)
        elif margin == 'channel':
            if self.datatype == 'epochs':
                non_margin_axes = 1
            elif self.datatype == 'tfr':
                non_margin_axes = (1, 2)
        elif margin == 'time-frequency':
            if self.datatype == 'tfr':
                non_margin_axes = 0
                
        # Compute marginal scores
        scores = [h.mean(axis=non_margin_axes) for h in hadamards]
        return scores
    
    def plot_marginal_brain_scores(self, lv_idx, margin, axes=None):
        
        if axes is None:
            if margin in ['channel', 'time-frequency']:
                f, axes = plt.subplots(ncols=len(self.labels))
            else:
                f, axes = plt.subplots()
        else:
            if margin in ['channel', 'time-frequency']:
                # TODO: more informative error msg
                assert axes.size == len(self.labels)
        
        scores = self.get_marginal_brain_scores(lv_idx=lv_idx, margin=margin)
        
        if margin in ['channel', 'time-frequency']:
            # Compute shared vlims
            vmax = max([np.abs(s).max() for s in scores])
        
        for idx in range(len(self.labels)):
            label = self.labels[idx]
            margin_scores = scores[idx]
            if margin == 'time':
                axes.plot(self.template.times, margin_scores, label=label)
            elif margin == 'frequency':
                axes.plot(self.template.freqs, margin_scores, label=label)
            elif margin == 'channel':
                ax = axes.flat[idx]
                mne.viz.plot_topomap(
                    data=margin_scores, pos=self.template.info, axes=ax,
                    show=False, vlim=(-vmax, vmax))
                ax.set_title(label)
            elif margin == 'time-frequency':
                ax = axes.flat[idx]
                self.template._data = np.stack([margin_scores]*len(self.template.info['ch_names']))
                self.template.plot(
                    combine='mean',
                    axes=ax,
                    show=False,
                    vlim=(-vmax, vmax))
                ax.set_title(label)
                    
        if margin == 'time':
            axes.set_xlabel('Time (s)')
            axes.set_ylabel('Temporal brain score')
            plt.legend()
        elif margin == 'frequency':
            axes.set_xlabel('Frequency (Hz)')
            axes.set_ylabel('Spectral brain score')
            plt.legend()
        
        plt.tight_layout()
 
def behavioral_pls(brain_data, cov_data, datatype, **kwargs):
    
    # TODO: validate size, type, etc of cov data; could be list
    
    template = utils.get_template(brain_data, datatype)
    
    datamat, submatrices = get_datamat(brain_data)
    if isinstance(cov_data, list):
        # stack
        cov_data = pd.concat(cov_data)
    
    pls_res = pyls.behavioral_pls(datamat, cov_data, **kwargs)
    
    labels = get_default_labels(**kwargs)
    if isinstance(cov_data, pd.DataFrame):
        cov_labels = list(cov_data.columns)
    else:
        cov_labels = ['cov %s' % i for i in range(cov_data.shape[1])]
    
    results = Results(template=template,
                      datatype=datatype,
                      submatrices=submatrices,
                      labels=labels,
                      cov_labels=cov_labels,
                      pls_type='beh',
                      pls_results=pls_res)
    
    return results

def meancentered_pls(data, datatype, labels=None, **kwargs):
    
    assert len(data) > 1
    
    # Figure out data type
    # TODO: make sure written datatype matches actual data
    # TODO: what if datatype is evoked?
    template = utils.get_template(data, datatype)
    
    # Default labels if none provided
    if labels == None:
        labels = get_default_labels(**kwargs)
    
    assert len(data) == len(labels)
    
    datamat, submatrices = get_datamat(data)
    
    pls_res = pyls.meancentered_pls(datamat, **kwargs)
    results = Results(template=template,
                      datatype=datatype,
                      submatrices=submatrices,
                      labels=labels,
                      pls_type='mc',
                      pls_results=pls_res)
    return results

def get_datamat(data):
    # Get datamat from list of data
    
    # Setup submatrices
    submatrices = []
    if type(data[0]) == list:
        # each element of list is a different individual/condition
        for obj in data:
            pass
        datamat = np.stack([d._data.flatten() for d in data])
    elif type(data) == list:
        # single individual analysis; list of data
        for obj in data:
            submatrix = np.stack([epoch.flatten() for epoch in obj._data])
            submatrices.append(submatrix)
        datamat = np.concat(submatrices)
    else:
        # single data object
        datamat = np.stack([epoch.flatten() for epoch in data._data])
        submatrices = [datamat] # hacky
    
    return datamat, submatrices

def get_default_labels(groups, n_cond=None, **kwargs):
    # TODO: edit to make use of group and cond!!!
    # labels = ['cond-%s' % (idx + 1) for idx in range(len(data))]
    labels = ['group %s cond %s' % (group_n, 0) for group_n in range(len(groups))]
    return labels

def _get_covariates(design, covariates, names):
    # Get covariates
    if design is not None:
        # Case 1: design is table, covariates is column names
        covariates = design[covariates].to_numpy()
    else:
        # Case 2: covariates is table, array, list
        covariate_array = np.array(covariates)
        # Ensure column vector
        if covariate_array.ndim == 1:
            covariate_array = np.expand_dims(covariate_array, 1)
    if names is None:
        if design is not None:
            names = covariates
        else:
            if 'columns' in dir(covariates):
                # dataframe
                names = covariates.columns.to_list()
            elif 'name' in dir(covariates):
                # series
                names = [covariates.name]
            else:
                names = ['cov%s' % i for i in range(covariate_array.shape[1])]
    return covariate_array, names