# GINet

<a href="https://geometricconvolutions.readthedocs.io/en/latest/">
<img src="https://badgen.net/badge/Read/the doc/blue" alt="doc"/>
</a>
<a href="https://github.com/WilsonGregory/GeometricConvolutions/blob/main/LICENSE">
<img src="https://badgen.net/badge/License/MIT/blue" alt="doc"/>
</a>

**Equivariant geometric convolutions for machine learning on tensor images**

This package implements the GeometricImageNet which allows for writing general functions from geometric images to geometric images. Also, with an easy restriction to group invariant CNN filters, we can write CNNs that are equivariant to those groups for geometric images.

See the paper for more details: https://arxiv.org/abs/2305.12585.

See our readthedocs for <a href="https://geometricconvolutions.readthedocs.io/en/latest/">full documentation</a>.

## Table of Contents

1. [Installation](#installation)
2. [Features](#features)
    1. [GeometricImage](#geometricimage)
    2. [MultiImage](#multiimage)
3. [Authors and Attribution](#authors-and-attribution)
4. [License](#license)

## Installation

- Install using pip: `pip install ginet`.
- Alternatively, you can install this repo as an editable install using pip.
  - Clone the repository `git clone https://github.com/WilsonGregory/GeometricConvolutions.git`
  - Navigate to the GeometricConvolutions directory `cd GeometricConvolutions`
  - Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
  - In order to run JAX on a GPU, you will likely need to follow some additional steps detailed in https://github.com/google/jax#installation. You will probably need to know your CUDA version, which can be found with `nvidia-smi` and/or `nvcc --version`.

## Features

### GeometricImage

The GeometricImage is the main concept of this package. We define a geometric image for dimension D, spatial dimensions, parity p, and tensor order k. To construct a geometric image, do: `image = GeometricImage(data, parity, D)`. Data is a jnp.array with the shape spatial dimensions followed by `(D,)*k)`.

### MultiImage

The MultiImage class allow us to group multiple images together that have the same dimension and spatial dimensions. MultiImage is a dictionary where the keys are (tensor order k, parity p) and the values are a image data block which has some number of initial axes followed by spatial axes and tensor axes. Common numbers of prior axes are 1 for channels, or 2 for batch followed by channels. The machine learning layers and models defined in this package expect a single axis for the channels, while the training code expects batch and then channels. You can easily construct MultiImages from images using the `from_images` function.

## Authors and Attribution
- **Wilson Gregory** (JHU)
- **Kaze W. K. Wong** (JHU)
- **David W. Hogg** (NYU) (MPIA) (Flatiron)
- **Soledad Villar** (JHU)

If you use this package in your own work, please cite the following:

```
@misc{gregory2024equivariantgeometricconvolutionsemulation,
      title={Equivariant geometric convolutions for emulation of dynamical systems}, 
      author={Wilson G. Gregory and David W. Hogg and Ben Blum-Smith and Maria Teresa Arias and Kaze W. K. Wong and Soledad Villar},
      year={2024},
      eprint={2305.12585},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2305.12585}, 
}
```

## License
Copyright 2022 the authors. All **text** (in `.txt` and `.tex` and `.bib` files) is licensed *All rights reserved*. All **code** (everything else) is licensed for use and reuse under the open-source *MIT License*. See the file `LICENSE` for more details of that.
