.. mne-plsc documentation master file, created by
   sphinx-quickstart on Tue Apr 14 22:46:52 2026.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

mne-plsc
========

``mne-plsc`` is a library for partial least squares correlation (PLSC) analysis of M/EEG data in Python, integrated with the MNE-Python library. The basic computations are performed by the `pyplsc <https://pyplsc.readthedocs.io/>`_ library, and the documentation of that library contains some background on the PLSC technique.

Installation
------------

``mne-plsc`` can be installed from PyPI via:

.. code-block::

   pip install mne-plsc

Quickstart
----------

The main functions for model fitting are ``fit_mc``, ``fit_beh``, and ``fit_within_beh``. These return objects whose methods can be used for permutation testing, cluster analysis, and visualization. The typical workflow would be:

1. Fit and visualize model
^^^^^^^^^^^^^^^^^^^^^^^^^^

Perform the initial decomposition and check the patterns of saliences.

.. code-block::

   from mne_plsc import fit_mc
   mod = fit_mc(epochs, condition)
   mod.plot_lv(0)

2. Permutation testing
^^^^^^^^^^^^^^^^^^^^^^

Evaluate which latent variables are significant.

.. code-block::

   mod.permute(1000)
   print(model.summary())

3. Cluster analysis
^^^^^^^^^^^^^^^^^^^

Perform bootstrap resampling to estimate brain salience z-scores, then cluster strong saliences (e.g., :math:`|z| > 2`).

.. code-block::

   mod.bootstrap(1000)
   mod.cluster(threshold=2)

4. Visualize cluster(s)
^^^^^^^^^^^^^^^^^^^^^^^

Examine the temporal/spectral/spatial distribution of the major clusters for a given set of brain saliences.

.. code-block::

   mod.plot_cluster_sizes(lv_idx=0)
   mod.plot_cluster(lv_idx=0, cluster_idx=0)

.. toctree::
   :maxdepth: 1
   :caption: Contents:

   examples
   fitting/fitting
   utils/utils
   classes/classes
