.. AIPyS documentation master file, created by
   sphinx-quickstart on Sat Mar  2 07:14:43 2024.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to AIPyS documentation!
================================
AIPyS - AI Powered Photoswitchable Genetic CRISPR Screen, Version 2

This project developed an AI Powered Photoswitchable Genetic CRISPR Screen, Version 2. A parametric object detection platform was developed, utilising Numpy, scikit-image, and scipy to accurately detect and segment single cells from microscopy images. Neural network models such as U-net and cGAN were used for segmentation, and a PyMC3 library was used to apply a Bayes' Logistic Model for the detection of subcellular phenotypes. Differential sgRNA abundance analysis was implemented using several Bayesian inference strategies. The program was published online using Plotly-Dash, allowing users to interact with and visualize the results.


.. toctree::
   :maxdepth: 2
   :caption: Contents:

   introduction
   installation
   deploying_aipys
   segmentation/index
   classification/index
   source/modules

Introduction
------------

Read about the concepts behind AIPyS, its capabilities, and how it can transform your Crispr screen analyses.

Installation
------------

Step-by-step guide on setting up AIPyS for your system, including requirements and dependencies.

Deploying_AIPyS
---------------

Comprehensive list and examples of Command Line Interface commands for AIPyS.

Segmentation
------------

Learn about the segmentation models and algorithms implemented in AIPyS.

Classification
--------------

Understand how AIPyS classifies data using Bayesian methods and Convolutional Neural Networks.

modules
-------

Please note that the documentation of all the modules used in this project is not complete yet. We are working diligently to provide comprehensive documentation for every module. Thank you for your patience.