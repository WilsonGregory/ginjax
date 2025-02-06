# Geometric Convolutions

This package implements the GeometricImageNet which allows for writing general functions from geometric images to geometric images. Also, with an easy restriction to group invariant CNN filters, we can write CNNs that are equivariant to those groups for geometric images.

See the paper for more details: https://arxiv.org/abs/2305.12585

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
    1. [Basic Features](#quick-start)
    2. [Learning Scalar Filters](#learning-scalar-filters)
3. [Features](#features)
    1. [GeometricImage](#geometricimage)
    2. [Layer and BatchLayer](#layer-and-batchlayer)
4. [Authors](#authors)
5. [License](#license)

## Installation

- Install using pip: `pip install geometricconvolutions`.
- Alternatively, you can install this repo as an editable install using pip.
  - Clone the repository `git clone https://github.com/WilsonGregory/GeometricConvolutions.git`
  - Navigate to the GeometricConvolutions directory `cd GeometricConvolutions`
  - Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)
  - In order to run JAX on a GPU, you will likely need to follow some additional steps detailed in https://github.com/google/jax#installation. You will probably need to know your CUDA version, which can be found with `nvidia-smi` and/or `nvcc --version`.

## Quick Start

### Basic Features
See the script `quick_start.py` for this example in code form.

First our imports. Geometric Convolutions is built in JAX. The majority of the model code resides in geometric.
```
import jax.numpy as jnp
import jax.random as random

import geometricconvolutions.geometric as geom
```

First we construct our image. Suppose you have some data that forms a 3 by 3 vector image, so N=3, D=2, and k=1. Currently only D=2 or D=3 images are valid, and the side lengths must all be equal. The parity is how the image responds when it is reflected. Normal images have parity 0, an image of pseudovectors like angular velocity will have parity 1.
```
key = random.PRNGKey(0)
key, subkey = random.split(key)

N = 3
D = 2
k = 1
parity = 0
data = random.normal(subkey, shape=((N,)*D + (D,)*k))
image = geom.GeometricImage(data, parity=0, D=2)
```

We can visualize this image with the plotting tools in utils. You will need to call matplotlib.pypolot.show() to display.
```
image.plot()
```

Now we can do various operations on this geometric image
```
image2 = geom.GeometricImage.fill(N, parity, D, fill=jnp.array([1,0])) # fill constructor, each pixel is fill

# pixel-wise addition
image + image2

# pixel-wise subtraction
image - image2

# pixel-wise tensor product
image * image2

# scalar multiplication
image * 3
```

We can also apply a group action on the image. First we generate all the operators for dimension D, then we apply one
```
operators = geom.make_all_operators(D)
print("Number of operators:", len(operators))
image.times_group_element(operators[1])
```

Now let us generate all 3 by 3 filters of tensor order k=0,1 and parity=0,1 that are invariant to the operators
```
invariant_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0,1],
    parities=[0,1],
    D=D,
    operators=operators,
    scale='one', #all the values of the filter are 1, can also 'normalize' so the norm of the tensor pixel is 1
    return_list=True,
)
print('Number of invariant filters N=3, k=0,1 parity=0,1:', len(invariant_filters))
```

Using these filters, we can perform convolutions on our image. Since the filters are invariant, the convolution
will be equivariant.
```
gg = operators[1] # one operator, a flip over the y-axis
ff_k0 = invariant_filters[1] # one filter, a non-trivial scalar filter
print(
    "Equivariant:",
    jnp.allclose(
        image.times_group_element(gg).convolve_with(ff_k0).data,
        image.convolve_with(ff_k0).times_group_element(gg).data,
        rtol=1e-2,
        atol=1e-2,
    ),
)
```

When convolving with filters that have tensor order > 0, the resulting image have tensor order img.k + filter.k
```
ff_k1 = invariant_filters[5]
print('image k:', image.k)
print('filter k:', ff_k1.k)
convolved_image = image.convolve_with(ff_k1)
print('convolved image k:', convolved_image.k)
```

After convolving, the image has tensor order 1+1=2 pixels. We can transpose the indices of the tensor:
```
convolved_image.transpose((1,0))
```

Since the tensor order is >= 2, we can perform a contraction on those indices which will reduce it to tensor order 0.
```
print('contracted image k:', convolved_image.contract(0,1).k)
```

### Learning Scalar Filters
Now we will have a simple example where we use GeometricConvolutions and JAX to learn scalar filters. See `scalar_example.py` for a python script of the example. First, the imports:
```
import time
import optax
from typing_extensions import Optional, Self

import jax
from jax import random
from jaxtyping import ArrayLike
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
```

First, lets define our inputs, `layer_X`. The models we define run on objects of the class `BatchLayer`. This object basically collects all the batches and channels of geometric images at different tensors and parities into a single object. To construct this, we pass a dictionary mapping (tensor_order,tensor_parity) to a block of data with axes (batch,channels,spatial_dimensions,tensor_dimensions). For this example, our images will be 2D, 64 x 64 scalar images.
```
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
num_images = 10

key, subkey = random.split(key)
layer_X = geom.BatchLayer({(0, 0): random.normal(subkey, shape=(num_images, 1) + (N,) * D)}, D)
```

Our filters will be 3x3 and they will be the invariant scalar filters only. There are 3 of these, and the first one is the identity. We use `get_invariant_filters` to get a layer of these filters.
```
M = 3  #filter image size
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[M], ks=[0], parities=[0], D=D, operators=group_actions)
```

Now let us define our target function, and then construct our target images Y. The target function will merely be convolving by the filter at index 1, then convolving by the filter at index 2.
```
def target_function(layer: geom.BatchLayer, conv_filter_a: jax.Array, conv_filter_b: jax.Array) -> geom.BatchLayer:
    convolved_data = geom.convolve(
        layer.D,
        geom.convolve(layer.D, layer[(0, 0)], conv_filter_a[None, None], layer.is_torus),
        conv_filter_b[None, None],
        layer.is_torus,
    )
    return geom.BatchLayer({(0, 0): convolved_data}, layer.D, layer.is_torus)

layer_y = target_function(layer_X, conv_filters[(0, 0)][1], conv_filters[(0, 0)][2])
```

Now we define our network that will learn this target function. We will just have it apply two convolutions in a row, using the invariant scalar filters. Our loss function will be the mean-squared error. The `ml.train` function expects a `map_and_loss` function that takes as input the model, the input layer, the target layer, and any auxilliary data, which we won't use in this instance.
```
class SimpleModel(eqx.Module):
    D: int
    net: list[ml.ConvContract]

    def __init__(
        self: Self,
        D: int,
        input_keys: tuple[tuple[ml.LayerKey, int]],
        output_keys: tuple[tuple[ml.LayerKey, int]],
        conv_filters: geom.Layer,
        key: ArrayLike,
    ):
        self.D = D
        key, subkey1, subkey2 = random.split(key, num=3)
        self.net = [
            ml.ConvContract(input_keys, output_keys, conv_filters, False, key=subkey1),
            ml.ConvContract(output_keys, output_keys, conv_filters, False, key=subkey2),
        ]

    def __call__(self: Self, x: geom.BatchLayer):
        for layer in self.net:
            x = layer(x)

        return x

def map_and_loss(
    model: eqx.Module,
    layer_x: geom.BatchLayer,
    layer_y: geom.BatchLayer,
    aux_data: Optional[eqx.nn.State] = None,
) -> float:
    return ml.smse_loss(layer_y, model(layer_x)), aux_data
```

Now we initialize our model, train it using the train function with a given optimizer for 500 epochs, and print the resulting weights.
```
key, subkey = random.split(key)
model = SimpleModel(D, layer_X.get_signature(), layer_y.get_signature(), conv_filters, subkey)

key, subkey = random.split(key)
trained_model, _, _, _ = ml.train(
    layer_X,
    layer_y,
    map_and_loss,
    model,
    subkey,
    ml.EpochStop(500, verbose=1),
    num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(trained_model.net[0].weights)
print(trained_model.net[1].weights)
```

This should print something like:
```
Epoch 50 Train: 0.0834250 Epoch time: 0.01208
Epoch 100 Train: 0.0006962 Epoch time: 0.01121
Epoch 150 Train: 0.0000310 Epoch time: 0.01187
Epoch 200 Train: 0.0000040 Epoch time: 0.01143
Epoch 250 Train: 0.0000011 Epoch time: 0.01129
Epoch 300 Train: 0.0000005 Epoch time: 0.01115
Epoch 350 Train: 0.0000003 Epoch time: 0.01112
Epoch 400 Train: 0.0000002 Epoch time: 0.01215
Epoch 450 Train: 0.0000002 Epoch time: 0.01202
Epoch 500 Train: 0.0000002 Epoch time: 0.01207
{(0, 0): {(0, 0): Array([[[3.7528062e-04, 9.0615535e-01, 1.1971369e-04]]], dtype=float32)}}
{(0, 0): {(0, 0): Array([[[-2.0696800e-05, -2.1163450e-04,  1.1035721e+00]]], dtype=float32)}}
 ```
and we can see that the first convolution is almost 1 for index 1, and the 2ond is almost 1 at index 2, as desired. Hooray!

## Features

### GeometricImage

The GeometricImage is the main concept of this package. We define a geometric image for dimension D, sidelength N, parity p, and tensor order k. Note that currently, all the sidelengths must be the same. To construct a geometric image, do: `image = GeometricImage(data, parity, D)`. Data is a jnp.array with the shape `((N,)*D + (D,)*k)`.

### Layer and BatchLayer

The Layer and BatchLayer classes allow us to group multiple images together that have the same dimension and sidelength. Layer is a dictionary where the keys are tensor order k, and the values are a image data block where the first index is the channel, then the remaining indices are the normal ones you would find in a geometric image. BatchLayer has the same structure, but the first index of the data image block is the batch, the second is the channel, and then the rest are the geometric image. You can easily construct Layers and BatchLayers from images using the `from_images` function.

## Authors
- **David W. Hogg** (NYU) (MPIA) (Flatiron)
- **Soledad Villar** (JHU)
- **Wilson Gregory** (JHU)

## License
Copyright 2022 the authors. All **text** (in `.txt` and `.tex` and `.bib` files) is licensed *All rights reserved*. All **code** (everything else) is licensed for use and reuse under the open-source *MIT License*. See the file `LICENSE` for more details of that.
