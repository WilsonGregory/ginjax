---
title: 'GeometricConvolutions: E(d)-Equivariant CNN for Tensor Images'
tags:
  - Python
  - Jax
  - machine learning
  - E(d)-equivariance
  - tensor images
  - Equinox
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
---

# Summary

Many data sets encountered in machine learning exhibit symmetries that can be exploited to improve performance, a technique known as equivariant machine learning. 
The classical example is image translation equivariance that is respected by convolutional neural networks [@lecun1989backpropagation]. 
For data sets in the physical sciences and other areas, we would also like equivariance to rotations and reflections. 
This Python package implements a convolutional neural network that is equivariant to translations, rotations of 90 degrees, and reflections. 
We implement this by _geometric convolutions_ [@gregory2024ginet] which uses tensor products and tensor contractions. 
This additionally enables us to perform functions on geometric images, or images where each pixel is a higher order tensor. 
These images appear as discretizations of fields in physics, such as velocity fields, vorticity fields, magnetic fields, polarization fields, and so on. 

This package includes basic functionality to create, manipulate, and plot geometric images using JAX [@jax2018github]. 
It also provides equivariant layers such as convolutions, activation functions, group norms, and others using the Equinox framework [kidger2021equinox]. 
These layers ingest a special data structure, the `MultiImage`, that allows combining geometric images or any tensor order or parity into a single model. 
Finally, the package provides full-fledged versions of popular models such as the UNet or ResNet to allow researchers to quickly train and test on their own data sets with standard tools in the JAX ecosystem.

# Statement of need

Let $G$ be a group with an action on vector spaces $X$ and $Y$. 
Let $f$ be a function from $X$ to $Y$. 
Then we say $f$ is $G$-equivariant if for all $g \in G$, $x \in X$, we have $f(g \cdot x) = g \cdot f(x)$. 
We will consider our group $G$ to be the _hyperoctahedral group_ of dimension $d$, denoted $G_{N,d}$. 
This is the group of translations, rotations, and reflections of a $d$-dimensional hypercube. 
This group will act on _geometric images_, that is images where every pixel is a scalar, vector, or higher order tensor. 

Lets consider some examples of geometric images. 
A scalar image is like a regular black and white image where each pixel is a single grayscale value. 
When we rotate a scalar image, the pixels move to their new location according to the rotation. We can have a color image by having multiple channels of a scalar image, for example 3 channels for a RGB image. 
A vector image could be something like an ocean current map where each pixel is a $d$-dimensional vector that shows where the water is flowing. 
When we rotate a vector image, not only do the pixels move to their new location, the vector-valued pixels themselves are also rotated. 
The behavior of the image under rotations and reflections is what distinguishes a vector image or a higher order tensor image from multiple channels of a scalar image.

To properly write functions geometric images, we have to keep track of the different types of images to make sure we respect the rotation properties of each. 
In the steerable cnns literature, [@cohen2016steerablecnns;@wweiler2021steerable], these are known as "steerable" spaces. 
A similar strategy for some tensor types can be implemented using Clifford Algebras as in [@brandstetter2023clifford]. 
This package implements geometric convolutions used in [@gregory2024ginet]. 
If $A$ is a geometric image of tensor order $k$ and $C$ is a geometric image of tensor order $k'$, then value of $A$ convolved with $C$ at pixel $\bar\imath$ is given by:
$$
(A \ast C)(\bar\imath) = \sum_{\bar a} A(\bar\imath - \bar a) \otimes C(\bar a) ~,
$$
where the sum is over all pixels $\bar a$ of $C$, and $\bar\imath - \bar a$ is the translation of $\bar\imath$ by $\bar a$. 
The result is a geometric image of tensor order $k+k'$. 
To produce geometric images of smaller tensor order, the tensor contraction can be applied to each pixel. 
Convolution and contraction are combined into a single operation to form linear layers. 
By restricting the convolution filters $C$ to rotation and reflection invariant filters, we can create linear layers which are $G_{N,d}$-equivariant.


# Acknowledgements



# References