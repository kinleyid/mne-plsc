
import mne
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import gridspec, cm, colors, patches, ticker
# import seaborn as sns
from nilearn import image, plotting

from mne.viz.evoked import _rgb, _plot_legend
from mne.viz.utils import _plot_masked_image

from . import utils

from pdb import set_trace

def _get_ax(ax=None):
    if ax is None:
        f, ax = plt.subplots(layout='constrained')
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
                    patches.Patch(color=cm.tab10(code), label=cat)
                    for code, cat in enumerate(sub_df['within'].cat.categories)
                ]
                curr_ax.legend(handles=handles)
            curr_ax.set_xlabel(None)
            curr_ax.set_ylabel(None)
        f.supxlabel('Design score')
        f.supylabel('Brain score')
    elif grouping == 'neither':
        df.plot.scatter(x='design_score',
                        y='data_score',
                        xlabel='Design score',
                        ylabel='Brain score',
                        ax=ax)
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
            patches.Patch(color=cm.tab10(code), label=cat)
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
        if index is None:
            pivoted = sub_df.pivot(columns=columns, values=pivot_values)
        else:
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
                    curr_ax.tick_params(axis='x',
                                        bottom=False,
                                        labelbottom=False)
        else:
            if grouping == 'neither':
                pivot_index = None
            else:
                pivot_index = grouping
            ax = _pivot_and_plot(df, index=pivot_index, columns='covariate', curr_ax=ax)
            ax.get_legend().set_title(None)
            # x ticks are not meaningful
            ax.tick_params(axis='x',
                           bottom=False,
                           labelbottom=False)
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
        y = ch_y[ch_idx]
        ax.plot(x,
                y,
                color=spatial_cols[ch_idx],
                linewidth=0.75,
                alpha=0.8)
        # Add scatter points when points become uncensored
        if np.ma.is_masked(y):
            padded = np.concatenate(([True], y.mask, [True]))
            diff = np.diff(padded.astype(int))
            mask_on = np.where(diff == -1)[0]
            mask_off = np.where(diff ==  1)[0] - 1
            changes = np.concatenate([mask_on, mask_off])
            ax.scatter(x[changes],
                       y[changes],
                       color=spatial_cols[ch_idx],
                       s=4)
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

def _check_log_scale(data):
    # Determine if log-scale
    ratios = data[1:] / data[:-1]
    if np.allclose(ratios, ratios[0]):
        scale = 'log'
    else:
        scale = 'linear'
    return scale

def tfr_image(template, data, cbar=True, vlabel=None, ax=None, vlim=None, ylabel='Frequency (Hz)', xlabel='Time (s)'):
    f, ax = _get_ax(ax)
    if vlim is None:
        vlim = tuple(np.array([-1, 1]) * np.abs(data).max())
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
        f.colorbar(im, ax=ax).set_label(vlabel)
    return f, ax

def chan_image(template, data, cbar=True, vlabel=None, vlim=None, ax=None):
    f, ax = _get_ax(ax)
    if vlim is None:
        vlim = tuple(np.array([-1, 1]) * np.abs(data).max())
    # Determine xdata
    if template.datatype == 'epo':
        xdata = template.times
        xlabel = 'Time (s)'
    elif template.datatype == 'spec':
        xdata = template.freqs
        xlabel = 'Frequency (Hz)'
    ydata = np.arange(template.info['nchan'])
    # Draw bounding box
    in_bounds = (~data.mask).sum(axis=0) > 0
    start, end = get_1d_lims(in_bounds)
    ax.axvline(xdata[start], c='k', ls=':')
    ax.axvline(xdata[end], c='k', ls=':')
    # Plot masked data
    im = ax.pcolormesh(xdata,
                       ydata,
                       data, # Masked array
                       cmap='RdBu_r',
                       vmin=vlim[0],
                       vmax=vlim[1])
    # Draw contours
    ax.contour(xdata,
               ydata,
               data.mask,
               levels=[0.5],
               corner_mask=False,
               antialiased=False,
               colors=['k'])
    ax.set(yticks=ydata)
    ax.set(yticklabels=template.info['ch_names'])
    ax.set_xlabel(xlabel)
    if cbar:
        f.colorbar(im, ax=ax).set_label(vlabel)
    return f, ax

def get_raster_data(template, data, xdim, ydim):
    # Get data
    dim_data = {
        'time': lambda: template.times,
        'freq': lambda: template.freqs,
        'chan': lambda: np.arange(template.info['nchan']),
        'vert': lambda: np.arange(sum(len(v) for v in template.vertices))}
    xdata = dim_data[xdim]()
    ydata = dim_data[ydim]()
    # Get labels
    dim_labels = {
        'time': 'Time (s)',
        'freq': 'Frequency (Hz)',
        'chan': None,
        'vert': 'Source index'}
    xlabel = dim_labels[xdim]
    ylabel = dim_labels[ydim]
    
    return xdata, ydata, xlabel, ylabel
    
    if template.ndim == 2:
        # One non-spatial dimension
        vdata = data
        if template.dimnames[1] == 'time':
            xdata = template.times
            xlabel = 'Time (s)'
        elif template.dimnames[1] == 'freq':
            xdata = template.freqs
            xlabel = 'Frequency (Hz)'
        if template.space == 'sensor':
            ydata = np.arange(template.info['nchan'])
            ylabel = None
        elif template.space == 'source':
            n_vert = sum(len(v) for v in template.vertices)
            ydata = np.arange(n_vert)
            ylabel = 'Source index'
    elif template.ndim == 3:
        vdata = data.mean(axis=-1)
        # Assume time-frequency
        xdata = template.times
        xlabel = 'Time (s)'
        ydata = template.freqs
        ylabel = 'Frequency (Hz)'
    data = (xdata, ydata, vdata)
    return data, xlabel, ylabel

def plot_raster(template, data, xdim, ydim, vlabel, ax):
    f, ax = _get_ax(ax)
    xdata, ydata, xlabel, ylabel = get_raster_data(template, data, xdim, ydim)
    out = _plot_raster_data(template, xdata, ydata, data, xlabel, ylabel, vlabel, ax)
    return out
   
def _plot_raster_data(template, xdata, ydata, data, xlabel, ylabel, vlabel, ax):
    vlim = np.abs(data).max()
    im = ax.pcolormesh(xdata, ydata, data,
                       cmap='RdBu_r',
                       vmin=-vlim,
                       vmax=vlim)
    # Set axis scales
    xscale = _check_log_scale(xdata)
    ax.set_xscale(xscale)
    yscale = _check_log_scale(ydata)
    ax.set_yscale(yscale)
    # Axis labels
    ax.set_xlabel(xlabel)
    if ylabel is None:
        # Channel labels
        ax.set_yticks(np.arange(template.info['nchan']))
        ax.set_yticklabels(template.info['ch_names'])
    else:
        ax.set_ylabel(ylabel)
    # Colorbar and label
    f = ax.figure
    f.colorbar(im, ax=ax).set_label(vlabel)
    return f, ax

def space_raster(template, data, cbar=True, vlabel=None, vlim=None, ax=None):
    f, ax = _get_ax(ax)
    if vlim is None:
        vlim = tuple(np.array([-1, 1]) * np.abs(data).max())
    xdata = template.times
    if template.space == 'sensor':
        ydata = np.arange(template.info['nchan'])
    elif template.space == 'source':
        n_vert = sum([len(v) for v in template.vertices])
        ydata = np.arange(n_vert)
        
    im = ax.pcolormesh(xdata,
                       ydata,
                       data, # Potentially masked array
                       cmap='RdBu_r',
                       vmin=vlim[0],
                       vmax=vlim[1])
    if np.ma.is_masked(data):
        # Draw contours
        ax.contour(xdata,
                   ydata,
                   data.mask,
                   levels=[0.5],
                   corner_mask=False,
                   antialiased=False,
                   colors=['k'])
    # Labels for y axis
    if template.space == 'sensor':
        ydata = np.arange(template.info['nchan'])
    elif template.space == 'source':
        ax.set_ylabel('Source index')
    # Labels for x axis
    nonspace_dim = template.dimnames[1]
    if nonspace_dim == 'time':
        ax.set_xlabel('Time (s)')
    elif nonspace_dim == 'freq':
        ax.set_xlabel('Frequency (Hz)')
    # Colorbar
    if cbar:
        f.colorbar(im, ax=ax).set_label(vlabel)
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
    ax.axhline(0, color='k', linestyle='--')
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
        ax.set_ylabel('Variance explained')
        ax.yaxis.set_major_formatter(ticker.PercentFormatter())
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
    x = np.arange(len(cluster_sizes))
    ax.plot(x, cluster_sizes)
    ax.scatter(x, cluster_sizes)
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

def plot_cluster_spatial(data, template, cluster, cluster_info, highlight, ax=None):
    f, ax = _get_ax(ax)
    # Get colorbar labels
    if cluster_info['which'] == 'saliences':
        which = 'salience'
    elif cluster_info['which'] == 'z-scores':
        which = 'bootstrap ratio (z score)'
    if highlight == 'peak':
        if template.ndim == 2:
            peak_dims = template.dimnames[1]
        else:
            peak_dims = 'time and frequency'
        vlabel = '%s at peak %s' % (which, peak_dims)
    elif highlight == 'extent':
        vlabel = 'mean %s in cluster extent' % which
    vlabel = vlabel.capitalize()
    if highlight == 'extent':
        # Get spatial data within cluster extent
        extent, _ = utils.get_cluster_extent(cluster['mask'])
        spatial_data = data[:, extent].mean(axis=-1)
        # Highlight all channels that are ever in the cluster
        spatial_mask = cluster['mask'].sum(axis=-1) > 0
    elif highlight == 'peak':
        # Get spatial data at non-spatial peak
        peak_coords = cluster['peak'][1:] # skip spatial dimension
        spatial_data = data[:, *peak_coords]
        # Highlight channels that are in the cluster at the peak
        spatial_mask = cluster['mask'][:, *peak_coords]
    if template.space == 'sensor':
        im, _ = mne.viz.plot_topomap(data=spatial_data,
                                     pos=template.info,
                                     axes=ax,
                                     mask=spatial_mask,
                                     show=False)
        # Colorbar
        cbar = ax.figure.colorbar(im, shrink=0.6)
        cbar.ax.set_ylabel(vlabel)
    elif template.datatype == 'surf-stc':
        spatial_data = spatial_data.reshape((-1, 1))
        spatial_data[~spatial_mask] = np.nan
        stc = mne.SourceEstimate(data=spatial_data,
                                 vertices=template.vertices,
                                 tmin=0, tstep=1,
                                 subject=template.subject) # TODO: does this make sense in general?
        stc.plot(subjects_dir='C:/Users/isaac/mne_data/MNE-sample-data/subjects',
                 time_viewer=False)
    elif template.datatype == 'vol-stc':
        spatial_data = spatial_data.reshape((-1, 1))
        spatial_data[~spatial_mask] = 0
        # Create volume
        stc = mne.VolSourceEstimate(
            data=spatial_data,
            vertices=template.vertices,
            tmin=0, tstep=1,
            subject=template.subject
        )
        vol = stc.as_volume(src=template.src)
        # Remove time dimension
        vol = image.index_img(vol, 0)
        # Display image
        plotting.plot_stat_map(vol,
                               bg_img=template.t1,
                               symmetric_cbar=True,
                               draw_cross=False,
                               axes=ax)
        # Label colorbar
        cbar_ax = f.get_axes()[-1]
        cbar_ax.set_ylabel(vlabel, color='white')
    return f, ax
    
def plot_cluster_butterfly(data, template, cluster, which, ythresh, highlight, ax=None):
    # Plot a cluster over non-channel margin(s)
    f, ax = _get_ax(ax)
    # Create masked data
    masked = np.ma.masked_array(data=data, mask=~cluster['mask'])
    # x axis is time or freq
    if template.datatype == 'epo':
        xdata = template.times
        xlabel = 'Time (s)'
    elif template.datatype == 'spec':
        xdata = template.freqs
        xlabel = 'Frequency (Hz)'
    else:
        raise ValueError('Butterfly plot not possible for datatype %s' % template.datatype)
    # Determine how to label data
    if which == 'saliences':
        ylabel = 'Salience'
    elif which == 'z-scores':
        ylabel = 'Bootstrap ratio (z score)'
    # Add highlighting first
    if highlight == 'extent':
        handle = plot_cluster_extent(xdata, cluster, ax)
    # Line plot with censor
    f, ax = channel_lineplot(xdata,
                             masked,
                             template.info,
                             ythresh=ythresh,
                             xlabel=xlabel,
                             ylabel=ylabel,
                             ax=ax)
    # Add peak after
    if highlight == 'peak':
        peak_t = xdata[cluster['peak'][1]]
        ax.axvline(peak_t, c='k', ls=':')
        peak_y = masked[cluster['peak']]
        handle = ax.scatter(peak_t, peak_y,
                            c='white', edgecolors='black',
                            label='Cluster peak')
    # Annotate highlighting
    if highlight != 'none':
        ax.legend(handles=[handle])
    return f, ax

def plot_cluster_raster(data, template, cluster, which, highlight, ax=None):
    masked = np.ma.MaskedArray(data=data, mask=~cluster['mask'])
    ydim, xdim = template.dimnames[-2:]
    if template.datatype == 'tfr':
        masked = masked.mean(axis=0)
        vlabel = 'Mean'
    else:
        vlabel = 'Raw'
    # Get data for raster plot
    xdata, ydata, xlabel, ylabel = get_raster_data(template, data, xdim, ydim)
    # Highlight extent behind cluster
    if highlight == 'extent':
        handle = plot_cluster_extent(xdata, cluster, ax, ydata)
    # Plot cluster
    f, ax = _plot_raster_data(template,
                              xdata=xdata,
                              ydata=ydata,
                              data=masked,
                              xlabel=xlabel,
                              ylabel=ylabel,
                              vlabel=vlabel,
                              ax=ax)
    # Draw contours
    ax.contour(xdata,
               ydata,
               masked.mask,
               levels=[0.5],
               corner_mask=False,
               antialiased=False,
               colors=['k'])
    # Highlight peak in front of cluster
    if highlight == 'peak':
        ypeak, xpeak = cluster['peak'][-2:]
        handle = ax.scatter(x=xdata[xpeak],
                            y=ydata[ypeak],
                            c='white', edgecolors='black',
                            label='Cluster peak')
    if highlight != 'none':
        ax.legend(handles=[handle],
                  loc='upper right')

"""
def plot_cluster_raster(data, template, cluster, which, highlight, ax=None):
    f, ax = _get_ax(ax)
    if template.ndim == 2: # 1 spatial, 1 non-spatial dimension
        # Determine how to label data
        if which == 'saliences':
            vlabel = 'Salience'
        elif which == 'z-scores':
            vlabel = 'Bootstrap ratio (z score)'
        # Determine non-spatial dimension
        nonspatial_dim = template.dimnames[1]
        if nonspatial_dim == 'time':
            xdata = template.times
            xlabel = 'Time (s)'
        elif nonspatial_dim == 'freq':
            xdata = template.freqs
            xlabel = 'Frequency (Hz)'
        # Mask
        masked = np.ma.MaskedArray(data=data, mask=~cluster['mask'])
        # Show extent
        if highlight == 'extent':
            plot_cluster_extent(xdata, cluster, ax)
        # Plot
        vlim = np.abs(data).max()
        xdata = template.times
        if template.space == 'sensor':
            ydata = ydata = np.arange(template.info['nchan'])
        elif template.space == 'source':
            ydata = np.arange(sum(len(verts) for verts in template.vertices))
        im = ax.pcolormesh(xdata,
                           ydata,
                           masked,
                           cmap='RdBu_r',
                           vmin=-vlim,
                           vmax=vlim)
        # Draw contours
        ax.contour(xdata,
                   ydata,
                   cluster['mask'],
                   levels=[0.5],
                   corner_mask=False,
                   antialiased=False,
                   linewidths=0.25,
                   colors=['k'])
        # Plot peak
        if highlight == 'peak':
            peak_ch = cluster['peak'][0]
            peak_t = xdata[cluster['peak'][1]]
            ax.scatter(peak_t, peak_ch,
                       c='white', edgecolors='black',
                       label='Cluster peak')
        # Annotate highlight
        if highlight != 'none':
            ax.legend()
        # Labels for y axis
        if template.space == 'sensor':
            ydata = np.arange(template.info['nchan'])
            ax.set(yticks=ydata)
            ax.set(yticklabels=template.info['ch_names'])
        elif template.space == 'source':
            ax.set_ylabel('Source index')
        # Labels for x axis
        nonspace_dim = template.dimnames[1]
        if nonspace_dim == 'time':
            ax.set_xlabel('Time (s)')
        elif nonspace_dim == 'freq':
            ax.set_xlabel('Frequency (Hz)')
        # Colorbar
        f.colorbar(im, ax=ax).set_label(vlabel)
    else: # 2 non-spatial dimensions (time and freq)
        # Compute average over spatial dimension
        set_trace()
        masked = np.ma.MaskedArray(data=data,
                                   mask=~cluster['mask'])
        tfr_data = np.array(masked.mean(axis=0))
        # Determine how to label data
        if which == 'saliences':
            vlabel = 'Mean salience in cluster'
        elif which == 'z-scores':
            vlabel = 'Mean bootstrap ratio (z score) in cluster extent'
        f, ax = tfr_image(template=template,
                          data=tfr_data,
                          vlabel=vlabel,
                          ax=ax)
    return f, ax
"""


def get_nonspatial_dim(template):
    nonspatial_dim = template.dimnames[1]
    if nonspatial_dim == 'time':
        xdata = template.times
        xlabel = 'Time (s)'
    elif nonspatial_dim == 'freq':
        xdata = template.freqs
        xlabel = 'Frequency (Hz)'
    return xdata, xlabel

def plot_cluster_distribution(template, cluster, highlight, ax=None):
    # Plot the distribution of the cluster over non-spatial axes
    f, ax = _get_ax(ax)
    if len(template.dimnames) == 2:
        n_in_clust = cluster['mask'].sum(0)
        # Data and label for non-spatial dimension
        xdata, xlabel = get_nonspatial_dim(template)
        if highlight == 'extent':
            handle = plot_cluster_extent(xdata, cluster, ax)
        # Plot distribution over non-spatial dimension
        ax.plot(xdata, n_in_clust)
        if highlight == 'peak':
            peak_x = xdata[cluster['peak'][1]]
            handle = ax.axvline(peak_x, c='k', ls=':',
                                label='Cluster peak')
        # Labels
        ax.set_xlabel(xlabel)
        spatial_dim = template.dimnames[0]
        if spatial_dim == 'vert':
            ylabel = 'N. vertices in cluster'
        else:
            ylabel = 'N. channels in cluster'
            ax.set_ylim((0, template.info['nchan']))
        ax.set_ylabel(ylabel)
        if highlight != 'none':
            ax.legend(handles=[handle])
    elif template.datatype == 'tfr':
        raise NotImplementedError()
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
                                 sharey=True,
                                 layout='constrained')
            for idx, (btwn, sub_df) in enumerate(df.groupby('between')):
                btwn_ax = ax[idx]
                btwn_ax.set_title(btwn)
                for _, row in sub_df.iterrows():
                    btwn_ax.plot(x, row['scores'], label=row['within'])
            ax[0].legend()
            f.supxlabel(xlabel)
            f.supylabel('Brain score')
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

def plot_cluster_extent(xdata, cluster, ax, ydata=None):
    in_extent, lim_pairs = utils.get_cluster_extent(cluster['mask'])
    if in_extent.ndim == 1:
        handle = ax.fill_between(xdata, 0, 1, where=in_extent,
                                 color='lightgray',
                                 transform=ax.get_xaxis_transform(), # fill whole y axis
                                 label='Cluster extent')
    else:
        cmap = colors.ListedColormap(['white', 'lightgray'])
        ax.pcolormesh(xdata,
                      ydata,
                      in_extent,
                      cmap=cmap)
        handle = patches.Patch(facecolor='lightgray',
                               edgecolor='none',
                               label='Cluster extent')
    return handle

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