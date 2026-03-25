
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec

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

def design_barplot(df, grouping, with_ci=False, ax=None):
    # df is the output of "contrast_to_df"
    f, ax = _get_ax(ax)
    if grouping == 'both':
        pivot_values = ['brain_score']
        if with_ci:
            pivot_values += ['L_CI', 'U_CI']
        pivoted = df.pivot(index='between', # x by between
                           columns='within', # colour by within
                           values=pivot_values)
        if with_ci:
            yerr = {}
            for col in pivoted['brain_score'].columns:
                yerr[col] = np.array([
                    pivoted['brain_score'][col] - pivoted['L_CI'][col],
                    pivoted['U_CI'][col] - pivoted['brain_score'][col],
                ])
        else:
            yerr = None
        ax = pivoted['brain_score'].plot.bar(yerr=yerr,
                                             ax=ax)
        ax.get_legend().set_title(None)
        # Create space for legend
        ylim = ax.get_ylim()
        ax.set_ylim((ylim[0], 1.5*ylim[1]))
    else:
        if with_ci:
            yerr = np.array([df['brain_score'] - df['L_CI'],
                             df['U_CI'] - df['brain_score']])
        else:
            yerr = None
        ax = df.plot.bar(x=grouping,
                         y='brain_score',
                         legend=False,
                         yerr=yerr,
                         ax=ax)
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
    _plot_legend(
        pos, colors=spatial_cols, axis=ax, bads=[], outlines=outlines, loc='upper left')
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

### For plotting clusters

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