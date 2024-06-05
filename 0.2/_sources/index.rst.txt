.. simbig documentation master file, created by
   sphinx-quickstart on Mon Mar 28 15:55:27 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

SIMulation-Based Inference of Galaxies
======================================


``SimBIG`` is a forward modeling framework for extracting cosmological information 
from the 3D spatial distribution of galaxies. 
It uses simulation-based inference (SBI) to perform highly efficient cosmological parameter 
inference using neural density estimation from machine learning.
``SimBIG`` enables us to leverage high-fidelity simulations that model the full details 
of the observed galaxy distribution and robustly analyze higher-order clustering
on small, non-linear, scales, beyond current standard analyses.

|simbig|
|br| *A schematic illustration of the* ``SimBIG`` *framework*

In |letter|_ we analyzed the galaxy power spectrum from the Sloan Digital Sky Survey-III 
|boss|_ and demonstrated that we can rigorously analyze galaxy clustering down to smaller
scales than ever before and extract more cosmological information than current
standard anlayses.
In upcoming papers (Hahn *et al.* in prep, Lemos *et al.* in prep, Régaldo-Saint Blancard
*et al.* in prep), we will present cosmological constraints from the ``SimBIG`` analyses 
of higher-order clustering.

.. _letter: https://www.pnas.org/doi/10.1073/pnas.2218810120
.. |letter| replace:: Hahn *et al.* (2023)

.. _boss: https://www.sdss3.org/surveys/boss.php
.. |boss| replace:: Baryon Oscillation Spectroscopic Survey (BOSS)


Forward Model
-------------

The ``SimBIG`` forward model is based on the high-resolution |quijote| 
*N*-body simulations that accurately model non-linear structure formation.
It uses a flexible state-of-the-art halo occupation model to model the 
galaxy-halo connection. 
It includes survey realism and observational systematics 
(*e.g.* survey geometry, masking, fiber collisions).
For details of the forward model, see |video|_ and |mocha|_.

.. _quijote: https://quijote-simulations.readthedocs.io/
.. |quijote| replace:: ``Quijote`` 

.. _video: https://youtube.com/playlist?list=PLQk8Faa2x0twK3fgs55ednnHD2vbIzo4z
.. |video| replace:: 3D comparison 

.. _mocha: https://ui.adsabs.harvard.edu/abs/2023JCAP...04..010H/abstract/
.. |mocha| replace:: Hahn *et al.* (2023b)

.. |simbig| image:: _static/graphics.jpg
   :align: middle 
   :height: 350

..
.. |br| raw:: html

   <br />

Contact
-------

.. toctree::
   :maxdepth: 2

   team
   papers
