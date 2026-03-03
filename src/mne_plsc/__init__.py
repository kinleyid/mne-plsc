
import mne
import pyls
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
import pandas as pd

from . import viz
from . import compute

from pdb import set_trace

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
    
    def get_template(self, lv_idx=0, which='bootstrap-ratios'):
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
        elif self.datatype == 'psd':
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
        elif self.datatype == 'psd':
            viz.plot_lv_psd(self.template, right_data, ax[1])
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
            elif self.datatype == 'psd':
                viz.plot_clust_nchan_psd(self.template, mask, nchan_ax)
            elif self.datatype == 'tfr':
                self.template._data = np.stack([mask.sum(axis=0)]*len(self.template.info['ch_names']))
                self.template.plot(combine='mean', axes=nchan_ax, show=False)
            
            # Plot topomap
            topo_ax = fig.add_subplot(gs[clust_idx*2 + 1, 1])
            if self.datatype == 'epochs':
                time_mask = mask.sum(axis=0) > 0
                topo_data = boot_rats[:, time_mask].mean(axis=1)
                ch_mask = mask.sum(axis=1) > 0
            elif self.datatype == 'psd':
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
            assert self.datatype in ['psd', 'tfr']
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
            if self.datatype == 'psd':
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
    
    template = get_template(brain_data, datatype)
    
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
    template = get_template(data, datatype)
    
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

def get_template(data, datatype):
    # Given a list of observations, get a single "template"
    # with info for plotting etc. later
    
    template = data[0]
    if datatype == 'epochs':
        # Epochs to Evoked
        template = mne.EvokedArray(
            data=np.zeros(template._data.shape[1:]),
            info=template.info,
            tmin=template.tmin)
    elif datatype == 'tfr':
        # EpochsTFR to AverageTFR
        template = template.average()
    elif datatype in ['spectrum', 'psd']:
        datatype = 'psd'
        template = template.average()
    else:
        pass
        # TODO: validate datatype
    
    return template

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
