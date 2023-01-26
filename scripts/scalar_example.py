import numpy as np
from jax import random
import time
import itertools as it
import math
import optax
from functools import partial

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, x, conv_filters):
    # A simple neural net that convolves with all combinations of each pair of conv_filters, then returns a linear combo
    layer_out = []
    for c1_idx, c2_idx in it.combinations_with_replacement(range(len(conv_filters)), 2):
        c1 = conv_filters[c1_idx]
        c2 = conv_filters[c2_idx]

        layer_out.append(x.convolve_with(c1).convolve_with(c2))

    return geom.linear_combination(layer_out, params)

def map_and_loss(params, x, y, conv_filters):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, conv_filters), y)

def target_function(x, conv_filters):
    return x.convolve_with(conv_filters[1]).convolve_with(conv_filters[2])

#Main
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 10

group_actions = geom.make_all_operators(D)
# start with basic 3x3 scalar filters
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)

key, subkey = random.split(key)
X = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=(num_images, N, N))]
Y = [target_function(x, conv_filters) for x in X]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + math.comb(len(conv_filters), 2),))

params = ml.train(
    X,
    Y,
    partial(map_and_loss, conv_filters=conv_filters),
    params,
    key,
    epochs=500,
    batch_size=num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(params)

