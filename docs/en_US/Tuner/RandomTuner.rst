Random Tuner
============

In `Random Search for Hyper-Parameter Optimization <http://www.jmlr.org/papers/volume13/bergstra12a/bergstra12a.pdf>`__ we show that Random Search might be surprisingly effective despite its simplicity.
We suggest using Random Search as a baseline when no knowledge about the prior distribution of hyper-parameters is available.

Usage
-----

Example Configuration

.. code-block:: yaml

   tuner:
     name: Random
     classArgs:
       seed: 100  # optional
