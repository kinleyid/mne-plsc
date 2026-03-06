
import mne
import numpy as np
from matplotlib import pyplot as plt

from pdb import set_trace

def design_barplot(y, labels=None, ci=None, ax=None, ylabel=None, facecolor='gray', edgecolor='black', **kwargs):
    # y must be (n,)
    # ci must be (2, n)
    # TODO: validate input shape
    if labels is None:
        labels = [str(x) for x in range(len(y))]
    if ci is None:
        yerr = None
    else:
        yerr = np.abs(ci - y)
    if ax is None:
        f, ax = plt.subplots()
    else:
        f = ax.figure
    ax.bar(labels, y,
           yerr=yerr,
           facecolor=facecolor,
           edgecolor=edgecolor,
           **kwargs)
    ax.tick_params('x', rotation=90)
    if ylabel:
        ax.set_ylabel(ylabel)
    return f, ax

### For plotting LVs

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