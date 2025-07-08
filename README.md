# ginjax

<a href="https://ginjax.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/WilsonGregory/ginjax/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

**Equivariant geometric convolutions for machine learning on tensor images**

The ginjax package implements the **G**eometric**I**mage**N**et in [jax](https://github.com/jax-ml/jax) (hence, ginjax) which allows for writing general functions from geometric images to geometric images. 
Also, with an easy restriction to group invariant CNN filters, we can write CNNs that are equivariant to those groups for geometric images.

See the paper for more details: [https://royalsocietypublishing.org/doi/full/10.1098/rsta.2024.0247?af=R](https://royalsocietypublishing.org/doi/full/10.1098/rsta.2024.0247?af=R).

See our readthedocs for <a href="https://ginjax.readthedocs.io/en/latest/">full documentation</a>.

## Table of Contents

1. [Installation](#installation)
    1. [Developer Installation](#developer-installation)
2. [Features](#features)
    1. [GeometricImage](#geometricimage)
    2. [MultiImage](#multiimage)
    3. [Equivariance](#equivariance)
3. [Authors and Attribution](#authors-and-attribution)
4. [License](#license)

## Installation

Install using pip: `pip install ginjax`.

### Developer Installation

To work on this package, install the repo as an editable install by doing the following:
- Clone the repository `git clone https://github.com/WilsonGregory/ginjax.git`
- Navigate to the ginjax directory `cd ginjax`
- Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
- In order to run JAX on a GPU, you will likely need to follow some additional steps detailed in https://github.com/google/jax#installation. You will probably need to know your CUDA version, which can be found with `nvidia-smi` and/or `nvcc --version`.
- In order to run the unit tests, install pytest `pip install pytest`. Then tests can be run by calling `pytest` from the package root directory.

## Features

### GeometricImage

The main concept of this package is an object called a `GeometricImage` which generalizes a normal image to include scalar images, vector images, or tensor images of any tensor order or parity.
For example, suppose we have a 16 by 16 vector image, that is a 2D image where every pixel is a vector in $\mathbb{R}^2$. 
You can think of this as 16 x 16 x 2 numbers, but the key observation is that the components of the vector in each pixel are not independent, as would be the case of two channels of a scalar image.
When you rotate a vector image 90 degrees to the right, the pixel locations rotate as well as the individual vectors in each pixel:
<p float="left">
  <img src="https://github.com/user-attachments/assets/1f6eadc4-8926-46ce-8c3e-e8b789aecddd" width="300" />
  <img src="https://github.com/user-attachments/assets/4a4ad1c4-bee5-459f-bf7e-f014b4d9fbe0" width="300" /> 
</p>

To construct a geometric image in ginjax, do the following:
```python
import jax.numpy as jnp
from ginjax.geometric import GeometricImage
D = 2 # 2-dimensional image, we could also do 3D.
data = jnp.arange(16*16*2).reshape((16,16,2))
parity = 0 # even parity, its a vector image not a pseudovector image
image = GeometricImage(data, parity, D)
```
Data is a jax numpy array with the shape spatial dimensions followed by `(D,)*k)`. Images default to being defined on the torus, aka periodic boundary conditions. 
When working with a metric that is not the flat Euclidean metric, you can define which tensor axes are contravariant (the default) or covariant.
Geometric images of a particular shape, tensor order, and parity form a vector space so we have the usual operations of addition, subtraction, and scalar multiplication:
```python
# some other image using fill constructor
image2 = geom.GeometricImage.fill(16, parity, D, fill=jnp.array([1,0])) 
added_image = image + image2 # addition
subtracted_image = image - image2 # subtraction
scaled_image = image * 3 # scalar multiplication
```
Additionally, if we have two images of the same spatial dimensions, but possibly different tensor orders or parities, we can multiply them where each pixel is the tensor product.
```python
image1 = GeometricImage(jnp.ones((3,3,D)), parity, D)
image2 = GeometricImage(jnp.ones((3,3,D,D)), parity, D)
image_product = image1 * image2 # data will be shape (3,3,2,2,2)
```
There are many more operations we can perform on geometric images including convolutions, contractions, rotations, norms, pooling, etc.
This example is continued on our [readthedocs](https://ginjax.readthedocs.io/en/latest/quickstart/) site.

### MultiImage

The GeometricImage class is useful for exploring and experimenting with individual geometric images; however, in machine learning contexts we typically need multiple channels and batches of images.
In addition, to build rotationally equivariant models, we may need to track multiple image types (scalar, vector, tensor, etc.) simultaneously. 
The MultiImage class allows us to do all of this at once, only making the assumption that the images are all in the same dimensional space (2D or 3D) and they all have the same spatial dimensions.
MultiImage is a dictionary where the keys are (tensor order k, parity p) and the values are a image data block which has some number of initial axes followed by spatial axes and tensor axes. Common numbers of prior axes are 1 for channels, or 2 for batch followed by channels. 
For example:
```python
D = 2 # dimension of the space
spatial_dims = (3,3) # image spatial dimensions

# first construct multi image with channels but not batch
multi_image1 = MultiImage(
    { 
        (0,0): jnp.ones((3,) + spatial_dims),  # 3 scalar channels
        (1,0): jnp.ones((1,) + spatial_dims + (D,)) # 1 vector channel
    }, 
    D,
)

# now construct multi image with batch dimension
batch = 5 # batch size
batch_multi_image2 = MultiImage(
    { 
        (1,1): jnp.ones((batch,1) + spatial_dims + (D,)),  # 1 pseudovector
        (2,0): jnp.ones((batch,2) + spatial_dims + (D,D)) # 2 2-tensor
    }, 
    D,
)
```
The machine learning layers and models defined in this package expect a single axis for the channels, while the training code expects batch and then channels. To use MultiImages for machine learning, see the [scalar example](https://ginjax.readthedocs.io/en/latest/examples/scalar_example/) and [gradient example](https://ginjax.readthedocs.io/en/latest/examples/gradient_example/). MultiImages can also be created with an associated metric tensor field.

### Equivariance

The main motivation of this package is for designing neural networks on geometric images that capture the symmetries of those geometric images.
Since we are working with discrete images, the symmetries are described by discrete groups such as translations, rotations of 90 degrees, reflections, or subgroups of these groups.
See [Group and Invariant Filters](https://ginjax.readthedocs.io/en/latest/group_operators/) for more details.
The scalars, vectors, and tensors of physics are equivariant to continuous rotations rather than just rotations of 90 degrees, but we do not consider them in this project to avoid voxelization issues.

We want our neural network as a function of geometric images to be _equivariant_ with respect to these groups.
If $G$ is a group with an action on vector spaces $X$ and $Y$ and $f$ is a function from $X$ to $Y$, then we say $f$ is $G$-equivariant if for all $g \in G$, $x \in X$, we have $f(g \cdot x) = g \cdot f(x)$. 
See [Math Background](https://ginjax.readthedocs.io/en/latest/math/) for more details.

## Authors and Attribution
- **Wilson Gregory** (JHU)
- **Kaze W. K. Wong** (JHU)
- **David W. Hogg** (NYU) (MPIA) (Flatiron)
- **Soledad Villar** (JHU)

If you use this package in your own work, please cite the following:

```
@article{doi:10.1098/rsta.2024.0247,
author = {Gregory, Wilson G.  and Hogg, David W.  and Blum-Smith, Ben  and Arias, Maria Teresa  and Wong, Kaze W. K.  and Villar, Soledad },
title = {Equivariant geometric convolutions for dynamical systems on vector and tensor images},
journal = {Philosophical Transactions of the Royal Society A: Mathematical, Physical and Engineering Sciences},
volume = {383},
number = {2298},
pages = {20240247},
year = {2025},
doi = {10.1098/rsta.2024.0247},

URL = {https://royalsocietypublishing.org/doi/abs/10.1098/rsta.2024.0247},
eprint = {https://royalsocietypublishing.org/doi/pdf/10.1098/rsta.2024.0247},
}
```

## License
Copyright 2022 the authors. All **text** (in `.txt` and `.tex` and `.bib` files) is licensed *All rights reserved*. All **code** (everything else) is licensed for use and reuse under the open-source *MIT License*. See the file `LICENSE` for more details of that.
