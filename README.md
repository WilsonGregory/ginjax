# Geometric Convolutions
group-equivariant convolutional deep learning

## Installation

- Install this repo as an editable install using pip.
  - Clone the repository `git clone https://github.com/WilsonGregory/GeometricConvolutions.git`
  - Navigate to the GeometricConvolutions directory `cd GeometricConvolutions`
  - Locally install the package `pip install -e .` (may have to use pip3 if your system has both python2 and python3 installed)

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
utils.plot_image(image)
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
    'Equivariant:',
    image.times_group_element(gg).convolve_with(ff_k0) == image.convolve_with(ff_k0).times_group_element(gg),
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
import numpy as np
import jax.numpy as jnp
from jax import random
import time
import itertools as it
import math
import optax
from functools import partial

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
```

Now lets define our images X and what filters we are going to use. Our image will be 2D, 64 x 64 scalar images. Our filters will be 3x3 and they will be the invariant scalar filters only. There are 3 of these, and the first one is the identity.
```
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 10

key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=(num_images, N, N))]

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)
```

Now let us define our target function, and then construct our target images Y. The target function will merely be convolving by the filter at index 1, then convolving by the filter at index 2.
```
def target_function(x, conv_filters):
    return x.convolve_with(conv_filters[1]).convolve_with(conv_filters[2])

Y = [target_function(x, conv_filters) for x in X]
```

Now let us define our network and loss function. For this toy example, we will make our task straightforward by making our network a linear combination of all the pairs of convolving by one filter from our set of three, then another filter from our set of three with replacement. In this fashion, our target function will be the 5th of 6 images. Our loss is simply the root mean square error loss (RMSE).
```
def net(params, x, D, is_torus, conv_filters):
    # A simple neural net that convolves with all combinations of each pair of conv_filters, then returns a linear combo
    conv_filters = jnp.stack([conv_filter.data for conv_filter in conv_filters])
    layer = []
    for i,j in it.combinations_with_replacement(range(len(conv_filters)), 2):
        layer.append(geom.convolve(D, geom.convolve(D, x, conv_filters[i], is_torus), conv_filters[j], is_torus))

    return geom.linear_combination(jnp.stack(layer), params)

def map_and_loss(params, x, y, conv_filters, D, is_torus):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, D, is_torus, conv_filters), y)
```

Now we initialize our params as random normal, then train our model using the `train` function from `ml.py`. Train takes the input data X, the target data Y, a map and loss function that takes arguments (params, x, y), the params array, a random key for doing the batches, the number of epochs to run, the batch size, and the desired optax optimizer.

```
key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + math.comb(len(conv_filters), 2),))

params = ml.train(
    X,
    Y,
    partial(map_and_loss, conv_filters=conv_filters, D=D, is_torus=True),
    params,
    key,
    epochs=500,
    batch_size=num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(params)
```

This should print something like:
```
Epoch 0: 557.6432495117188
Epoch 49: 28.267072677612305
Epoch 99: 12.66004753112793
Epoch 149: 2.608067274093628
Epoch 199: 1.395210862159729
Epoch 249: 0.4147546887397766
Epoch 299: 0.6743659973144531
Epoch 349: 0.23962187767028809
Epoch 399: 0.21458478271961212
Epoch 449: 0.11292069405317307
Epoch 499: 0.036848485469818115
[7.6919132e-05 6.8303452e-06 7.5942320e-05 7.7729273e-05 1.0000064e+00
 7.7100434e-05]
 ```
 and we can see that the 5th parameter is 1 and all others are tiny. Hooray!


## Authors
- **David W. Hogg** (NYU) (MPIA) (Flatiron)
- **Soledad Villar** (JHU)
- **Wilson Gregory** (JHU)

## License
Copyright 2022 the authors. All **text** (in `.txt` and `.tex` and `.bib` files) is licensed *All rights reserved*. All **code** (everything else) is licensed for use and reuse under the open-source *MIT License*. See the file `LICENSE` for more details of that.

## Projects
- Make nonlinear, group-equivariant, geometric convolution operators.
- Apply these to cosmological-simulation problems, perhaps.
