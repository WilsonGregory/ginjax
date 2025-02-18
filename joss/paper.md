---
title: 'GeometricConvolutions: E(d)-Equivariant CNN for Tensor Images'
tags:
  - Python
  - Jax
  - machine learning
  - E(d)-equivariance
  - tensor images
authors:
  - name: Wilson G. Gregory
    orcid: 0000-0002-5511-0683
    corresponding: true
    affiliation: 1 
  - name: Kaze W. K. Wong
    affiliation: 1
affiliations:
 - name: Department of Applied Mathematics and Statistics, Johns Hopkins University, Baltimore, MD, USA
   index: 1
date: 18 February 2025
bibliography: paper.bib

# Summary

Many data sets encountered in machine learning exhibit symmetries that can be exploited to improve performance. The classic example is image problems that have translation symmetries. An image classification may be the same regardless of where the object of interest is in the image, in which case we say the classifier is translation invariant. Similarly, if we are doing image segmentation and we shift the image over, then the segmentation is shifted as well, a property called translation equivariance. Image convolution preserves these properties. If we desire further equivariance to other groups such as rotation and reflection equivariance, then we require additional techniques.

This Python package implements a convolutional neural network that is equivariant to translations, rotations of 90 degrees, and reflections. We implement this by _geometric convolutions_ which use tensor products and tensor contractions. This additionally enables us to perform functions on geometric images, or images where each pixel is a higher order tensor. This is particularly useful for problems in the physical sciences, such as those involving velocity fields, vorticity fields, magnetic fields, polarization fields, and so on. This package includes basic functionality to create, manipulate, and plot geometric images, as well as machine learning tools to build and train networks which process geometric images using JAX [@jax2018github].

# Statement of need

tbd



# Acknowledgements



# References