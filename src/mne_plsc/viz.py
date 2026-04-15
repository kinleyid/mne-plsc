
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec
from matplotlib import cm, colors
import matplotlib.patches as mpatches
# import seaborn as sns

from mne.viz.evoked import _rgb, _plot_legend
from mne.viz.utils import _plot_masked_image

from . import utils

from pdb import set_trace

def _get_ax(ax=None):
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure
    return f, ax

def _subdivide_ax(ax, nrows=1, ncols=1, sharex=False, sharey=False):
    gs = gridspec.GridSpecFromSubplotSpec(nrows=nrows,
                                          ncols=ncols,
                                          subplot_spec=ax.get_subplotspec())
    f = ax.figure
    ax.remove()
    ax0 = f.add_subplot(gs[0])
    if sharex:
        sharex = ax0
    else:
        sharex = None
    if sharey:
        sharey = ax0
    else:
        sharey = None
    axes = [f.add_subplot(gs[i],
                                  sharex=ax0,
                                  sharey=ax0)
            for i in range(1, nrows*ncols)]
    ax = np.stack([ax0] + axes)
    ax = ax.reshape((nrows, ncols))
    return ax

def score_scatterplot(df, grouping, ax=None):
    f, ax = _get_ax(ax)
    if grouping == 'both':
        ax = _subdivide_ax(ax,
                           ncols=df['between'].nunique(),
                           sharex=True,
                           sharey=True)
        for ax_idx, (group_name, sub_df) in enumerate(df.groupby('between')):
            curr_ax = ax[0, ax_idx]
            col = sub_df['within'].cat.codes.map(cm.tab10)
            sub_df.plot.scatter(x='design_score',
                                y='data_score',
                                c=col,
                                ax=curr_ax)
            curr_ax.set_title(group_name)
            if ax_idx == 0:
                # Add legend manually
                handles = [
                    mpatches.Patch(color=cm.tab10(code), label=cat)
                    for code, cat in enumerate(sub_df['within'].cat.categories)
                ]
                curr_ax.legend(handles=handles)
            curr_ax.set_xlabel(None)
            curr_ax.set_ylabel(None)
        f.supxlabel('Design score')
        f.supylabel('Brain score')
        plt.tight_layout()
    else:
        col = df[grouping].cat.codes.map(cm.tab10)
        df.plot.scatter(x='design_score',
                        y='data_score',
                        c=col,
                        xlabel='Design score',
                        ylabel='Brain score',
                        ax=ax)
        # Add legend manually
        handles = [
            mpatches.Patch(color=cm.tab10(code), label=cat)
            for code, cat in enumerate(df[grouping].cat.categories)
        ]
        ax.legend(handles=handles)
    return f, ax

def boot_stat_barplot(df, boot_stat, grouping, with_ci=False, ax=None):
    f, ax = _get_ax(ax)

    def _compute_yerr(pivoted_stat, pivoted_l, pivoted_u):
        if not with_ci:
            return None
        return {col: np.array([pivoted_stat[col] - pivoted_l[col],
                               pivoted_u[col] - pivoted_stat[col]])
                for col in pivoted_stat.columns}

    def _pivot_and_plot(sub_df, index, columns, curr_ax):
        pivot_values = ['stat'] + (['L_CI', 'U_CI'] if with_ci else [])
        pivoted = sub_df.pivot(index=index, columns=columns, values=pivot_values)
        yerr = _compute_yerr(pivoted['stat'], pivoted.get('L_CI', {}), pivoted.get('U_CI', {})) if with_ci else None
        pivoted['stat'].plot.bar(yerr=yerr, ax=curr_ax)
        curr_ax.set_xlabel(None)
        return curr_ax

    if 'covariate' in df:
        # Beh PLS
        if grouping == 'both':
            groups = list(df.groupby('between'))
            ax = _subdivide_ax(ax, nrows=len(groups))
            for ax_idx, (group, sub_df) in enumerate(groups):
                curr_ax = ax[ax_idx, 0]
                curr_ax.set_title(group)
                _pivot_and_plot(sub_df, index='within', columns='covariate', curr_ax=curr_ax)
                legend = curr_ax.get_legend()
                # Show legend only for first axis
                if ax_idx == 0:
                    legend.set_title(None)
                else:
                    legend.remove()
                # Show x ticks only for last axis
                if ax_idx < (len(groups) - 1):
                    curr_ax.tick_params(axis='x', labelbottom=False)
        else:
            ax = _pivot_and_plot(df, index=grouping, columns='covariate', curr_ax=ax)
            ax.get_legend().set_title(None)
        f.supylabel('Correlation with brain score')
    else:
        # MC PLS
        if grouping == 'both':
            ax = _pivot_and_plot(df, index='between', columns='within', curr_ax=ax)
            ax.get_legend().set_title(None)
            ylim = ax.get_ylim()
            ax.set_ylim((ylim[0], 1.5 * ylim[1]))
        else:
            yerr = np.array([df['stat'] - df['L_CI'], df['U_CI'] - df['stat']]) if with_ci else None
            df.plot.bar(x=grouping, y='stat', legend=False, yerr=yerr, ax=ax)
            ax.set_xlabel(None)
        ax.set_ylabel('Brain score')

    return f, ax

def channel_lineplot(x, ch_y, info, ax=None, xlabel=None, ylabel=None, ythresh=None):
    f, ax = _get_ax(ax)
    ax.axhline(0, color='k')
    if ythresh is not None:
        # Add y threshold first
        ax.axhline(-ythresh, c='k', ls=':')
        ax.axhline(ythresh, c='k', ls=':')
    spatial_cols = get_spatial_colours(info)
    # Plot lines
    for ch_idx in range(len(info['chs'])):
        ax.plot(x,
                ch_y[ch_idx],
                color=spatial_cols[ch_idx],
                linewidth=0.75,
                alpha=0.8)
    # Add space for sensor legend
    ylim = ax.get_ylim()
    ax.set_ylim((ylim[0], 1.6*ylim[1]))
    # Show sensor legend
    pos, outlines = mne.viz.evoked._get_pos_outlines(
        info, picks=range(len(info['chs'])), sphere=None)
    _plot_legend(pos,
                 colors=spatial_cols,
                 axis=ax,
                 bads=[],
                 outlines=outlines,
                 loc='upper left')
    # Axis labels
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Ensure full x range is displayed
    ax.set_xlim((x[0], x[-1]))
    return f, ax

def tfr_image(template, data, cbar=True, clabel=None, ax=None, vlim=None, ylabel='Frequency (Hz)', xlabel='Time (s)'):
    if vlim is None:
        vlim = tuple(np.array([-1, 1]) * np.abs(data).max())
    f, ax = _get_ax(ax)
    # Determine if log-scale
    ratios = template.freqs[1:] / template.freqs[:-1]
    if np.allclose(ratios, ratios[0]):
        yscale = 'log'
    else:
        yscale = 'linear'
    im = ax.pcolormesh(template.times,
                       template.freqs,
                       data,
                       cmap='RdBu_r',
                       vmin=vlim[0],
                       vmax=vlim[1])
    ax.set_yscale(yscale)
    freq_landmarks = np.array([1, 4, 8, 13, 20, 30, 40, 80])
    ylims = ax.get_ylim()
    in_range = (freq_landmarks > ylims[0]) & (freq_landmarks < ylims[1])
    freq_landmarks = freq_landmarks[in_range]
    ax.set_yticks(freq_landmarks)
    ax.set_yticklabels([str(flm) for flm in freq_landmarks])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if cbar:
        f.colorbar(im, ax=ax).set_label(clabel)
    return f, ax

def get_spatial_colours(info):
    locs3d = np.array([ch['loc'][:3] for ch in info['chs']])
    x, y, z = locs3d.T
    return _rgb(x, y, z)

def plot_lv_epochs(template, data, axes):
    tmp = template.copy()
    tmp._data = 1e-6 * np.reshape(data, tmp._data.shape)
    # Butterfly plot
    tmp.plot(axes=axes, show=False)
    # tmp.plot_joint(show=True)
    # Remove title
    axes.set_title(None)
    # Remove Nave text
    axes.texts[1].remove()

def plot_lv_tfr(template, data, axes):
    tmp = template.copy()
    tmp._data = np.reshape(data, tmp.shape)
    tmp.plot(axes=axes, combine='mean', show=False)
    # TODO: options for log transforming

def plot_lv_psd(template, data, axes):
    tmp = template.copy()
    tmp._data = 1e-12 * np.reshape(data, tmp.shape)
    tmp.plot(axes=axes, dB=False, amplitude=False, show=False)
    axes.grid(False)
    axes.set_title(None)
    # Remove vertical lines
    # axes.lines[0].remove()
    # axes.lines[1].remove()
    axes.set_xlabel('Frequency (Hz)')
    # TODO: options for log transforming

def scree(singular_vals, which, rank, null_dist=None, null_percentile=95, ax=None):
    f, ax = _get_ax(ax)
    if which == 'pct-variance':
        total_variance = np.sum(singular_vals**2)
        singular_vals = 100*singular_vals**2/total_variance
    # Plot non-null 
    ax.scatter(x=np.arange(rank),
               y=singular_vals[:rank],
               c='black',
               label='Observed')
    # Plot null
    ax.scatter(x=np.arange(rank, len(singular_vals)),
               y=singular_vals[rank:],
               c='black',
               marker='x')
    ax.set_xlabel('Latent variable pair index')
    if which == 'pct-variance':
        ax.set_ylabel('Percent variance explained')
    elif which == 'singular-val':
        ax.set_ylabel('Singular value')
    # Plot permutation distribution
    if null_dist is not None:
        if which == 'pct-variance':
            null_dist = 100*null_dist**2/total_variance
        mins = null_dist.min(axis=0)
        maxs = np.percentile(null_dist, null_percentile, axis=0)
        for i in range(rank):
            if i == 0:
                label = 'Null distribution\n(up to %sth\npercentile)' % null_percentile
            else:
                label = '_nolegend_'
            ax.plot([i]*2, [mins[i], maxs[i]], 'r', label=label)
        ax.legend()
    ax.set_xticks(np.arange(len(singular_vals)))
    ax.set_xticklabels(np.arange(len(singular_vals)))
    return f, ax

### For plotting clusters

def plot_cluster_sizes(cluster_sizes, size_measure='pct-strong', logx=False, ax=None):
    f, ax = _get_ax(ax)
    ax.plot(cluster_sizes)
    ax.set_xlabel('Cluster index')
    if size_measure == 'absolute':
        ax.set_ylabel('Cluster size (n. neural variables)')
    elif size_measure == 'pct-strong':
        ax.set_ylabel('Cluster size (% of strong saliences)')
    elif size_measure == 'pct-total':
        ax.set_ylabel('Cluster size (% of neural variables)')
    if logx:
        ax.set_xscale('log')
    return f, ax

def plot_cluster(data, template, cluster, cluster_info, non_chan_plot, ax=None):
    f, ax = _get_ax(ax)
    data = data.reshape(template.shape)
    # Go from linear indices to nd mask
    mask = np.zeros(template.shape, dtype=np.bool)
    mask.flat[cluster['idx']] = True
    # Divide into sub-axes
    sub_gs = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=ax.get_subplotspec())
    ax.remove()
    # Left panel: non-channel margin(s)
    ax_left = f.add_subplot(sub_gs[0])
    if non_chan_plot == 'masked-data':
        if cluster_info['which'] == 'saliences':
            ylabel = 'Salience'
        elif cluster_info['which'] == 'bootstrap-ratios':
            ylabel = 'Bootstrap ratio'
        f, ax_left = plot_cluster_nonchan_margin(data,
                                                 template,
                                                 mask,
                                                 cluster_info,
                                                 ylabel=ylabel,
                                                 ax=ax_left)
    elif non_chan_plot == 'density':
        plot_cluster_distribution(template,
                                  mask,
                                  ax=ax_left)
    # Right panel: topoplot
    ax_right = f.add_subplot(sub_gs[1])
    non_chan_axes = utils.get_non_margin_axes('chan', template.datatype)
    ch_mask = mask.sum(axis=non_chan_axes) > 0
    non_chan_mask = mask.sum(axis=0) > 0
    topo_data = data[:, non_chan_mask].mean(axis=1) # 1 or non_chan_axes?
    im, _ = mne.viz.plot_topomap(topo_data,
                                 template.info,
                                 axes=ax_right,
                                 mask=ch_mask, show=False)
    # Colorbar
    if cluster_info['which'] == 'saliences':
        clabel = 'Mean salience in cluster'
    elif cluster_info['which'] == 'bootstrap-ratios':
        clabel = 'Mean bootstrap ratio in cluster'
    cbar = ax_right.figure.colorbar(im, shrink=0.6)
    cbar.ax.set_ylabel(clabel)
    plt.tight_layout()
    return f, (ax_left, ax_right)

def plot_cluster_nonchan_margin(data, template, mask, cluster_info, ylabel=None, ax=None):
    # Plot a cluster over non-channel margin(s)
    f, ax = _get_ax(ax)
    # Censor out-of-cluster data
    masked = data.copy()
    masked[~mask] = np.nan
    if template.datatype in ['epo', 'spec']:
        # Line plot---x axis is time or freq
        if template.datatype == 'epo':
            x = template.times
            xlabel = 'Time (s)'
        elif template.datatype == 'spec':
            x = template.freqs
            xlabel = 'Frequency (Hz)'
        # Line plot with censor
        channel_lineplot(x,
                         masked,
                         template.info,
                         ythresh=cluster_info['threshold'],
                         xlabel=xlabel,
                         ylabel=ylabel,
                         ax=ax)
        # Draw bounding box
        n_chan = mask.sum(axis=0)
        start, end = get_1d_lims(n_chan > 0)
        ax.axvline(x[start], c='k', ls=':')
        ax.axvline(x[end], c='k', ls=':')
    elif template.datatype == 'tfr':
        # TFR image, but masked
        masked = np.ma.MaskedArray(data=data, mask=~mask)
        tfr_data = np.array(masked.mean(axis=0))
        if cluster_info['which'] == 'saliences':
            clabel = 'Mean salience'
        elif cluster_info['which'] == 'bootstrap-ratios':
            clabel = 'Mean bootstrap ratio'
        tfr_image(template, tfr_data, clabel=clabel, ax=ax)
        
    return f, ax

def get_1d_lims(bool_array):
    diff = np.diff(bool_array.astype(int))
    start = np.where(diff == 1)[0] + 1
    end = np.where(diff == -1)[0]
    if bool_array[0]:
        start = np.r_[0, start]
    if bool_array[-1]:
        end = np.r_[end, len(bool_array) - 1]
    return start, end

def plot_cluster_distribution(template, mask, ax=None):
    # Plot the distribution of the cluster over non-channel axes
    f, ax = _get_ax(ax)
    if template.datatype in ['epo', 'spec']:
        # Line plot---x axis is time or freq
        n_chan = mask.sum(axis=0)
        if template.datatype == 'epo':
            x = template.times
            xlabel = 'Time (s)'
        elif template.datatype == 'spec':
            x = template.freqs
            xlabel = 'Frequency (Hz)'
        ax.plot(x, n_chan)
        # Draw limits
        start, end = get_1d_lims(n_chan > 0)
        ax.plot(x[[start, start]], [0, template.info['nchan']], 'k:')
        ax.plot(x[[end, end]], [0, template.info['nchan']], 'k:')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Channels in cluster')
        ax.set_ylim((0, len(template.info['chs'])))
    elif template.datatype == 'tfr':
        # TODO: implement
        pass
    return f, ax

def plot_marginal_brain_scores(scores, margin, labels, template, grouping, ax=None):
    # Combine labels and scores into one dataframe
    df = labels.assign(scores=scores)
    df = df.sort_values(by=['between', 'within'])
    if margin in ['time', 'freq']:
        if margin == 'time':
            x = template.times
            xlabel = 'Time (s)'
        elif margin == 'freq':
            x = template.freqs
            xlabel = 'Frequency (Hz)'
        if grouping == 'both':
            # Line plots faceted by between condition and coloured by within condition
            f, ax = plt.subplots(nrows=df['between'].nunique(),
                                 sharex=True,
                                 sharey=True)
            for idx, (btwn, sub_df) in enumerate(df.groupby('between')):
                btwn_ax = ax[idx]
                btwn_ax.set_title(btwn)
                for _, row in sub_df.iterrows():
                    btwn_ax.plot(x, row['scores'], label=row['within'])
            ax[0].legend()
            f.supxlabel(xlabel)
            f.supylabel('Brain score')
            plt.tight_layout()
        else:
            # Line plots coloured by condition
            f, ax = plt.subplots()
            for _, row in df.iterrows():
                ax.plot(x, row['scores'], label=row[grouping])
            ax.legend()
            ax.set_ylabel('Brain score')
            ax.set_xlabel(xlabel)
    elif margin in ['chan', 'time-freq']:
        vlim = np.abs(np.stack(df['scores'])).max()
        # Set up axes---separate axis per condition
        if grouping == 'both':
            f, ax = plt.subplots(nrows=df['between'].nunique(),
                                 ncols=df['within'].nunique(),
                                 sharex=True, sharey=True)
        else:
            f, ax = plt.subplots(ncols=df[grouping].nunique(),
                                 sharex=True, sharey=True)
        # Plots
        for idx, row in df.iterrows():
            curr_ax = ax.flat[idx]
            if margin == 'chan':
                mne.viz.plot_topomap(
                    data=row['scores'],
                    pos=template.info,
                    vlim=(-vlim, vlim),
                    axes=curr_ax,
                    show=False)
            elif margin == 'time-freq':
                tfr_image(template,
                          scores[idx],
                          ax=curr_ax,
                          cbar=False,
                          vlim=(-vlim, vlim),
                          xlabel=None,
                          ylabel=None)
        # Shared colour bar
        cmap = cm.RdBu_r
        norm = colors.Normalize(vmin=-vlim, vmax=vlim)
        sm = cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = f.colorbar(sm, ax=ax, shrink=0.8)
        cbar.set_label('Brain score')
        # Label x and y axes
        if margin == 'time-freq':
            f.supxlabel('Time (s)')
            f.supylabel('Frequency (Hz)')
        # Label columns
        if grouping == 'both':
            top_axes = ax[0]
            row_labels = df['within'].cat.categories
        else:
            top_axes = ax
            row_labels = df[grouping].cat.categories
        for curr_ax, label in zip(top_axes, row_labels):
            curr_ax.set_title(label)
        if grouping == 'both':
            # Label rows
            for curr_ax, label in zip(ax[:, -1], df['between'].cat.categories):
                curr_ax.yaxis.set_label_position('right')
                curr_ax.annotate(label,
                                 xy=(0, 0.5),
                                 # xytext=(-curr_ax.yaxis.labelpad - 24, 0),
                                 xytext=(curr_ax.yaxis.labelpad, 5),
                                 xycoords=curr_ax.yaxis.label,
                                 textcoords="offset points",
                                 fontsize=12, ha="left", va="center")
    return f, ax

def plot_clust_nchan_epochs(template, mask, axes):
    n_chan = mask.sum(axis=0)
    axes.plot(template.times[[0, -1]], [len(template.info['ch_names'])]*2, 'k:')
    axes.plot(template.times, n_chan, 'k')
    axes.set_xlabel('Time (s)')
    axes.set_ylabel('Channels in cluster')

def plot_clust_nchan_psd(template, mask, axes):
    n_chan = mask.sum(axis=0)
    axes.plot(template.freqs[[0, -1]], [len(template.info['ch_names'])]*2, 'k:')
    axes.plot(template.freqs, n_chan, 'k')
    axes.set_xlabel('Frequency (Hz)')
    axes.set_ylabel('Channels in cluster')

def plot_topomap(template, data, ch_mask, axes, top_ppn=0.2):
    
    out = mne.viz.plot_topomap(
        data, template.info,
        axes=axes, mask=ch_mask, show=False)
    
    return out