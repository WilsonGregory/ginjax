import jax.numpy as jnp
from jax import random, vmap
import time
import itertools as it
import math
import optax
from functools import partial

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, layer, conv_filters):
    # A simple neural net that convolves with all combinations of each pair of conv_filters, 
    # then returns a linear combo
    out_layer = layer.empty()
    for i,j in it.combinations_with_replacement(range(len(conv_filters)), 2):
        convolved_layer = out_layer.__class__(
            { 0: target_function(layer, conv_filters[i].data, conv_filters[j].data) },
            layer.D,
            layer.is_torus,
        )
        out_layer = out_layer + convolved_layer

    return geom.Layer({ 0: geom.linear_combination(out_layer[0], params) }, layer.D, layer.is_torus)

def map_and_loss(params, x, y, key, train, conv_filters):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, conv_filters)[0], y[0])

def target_function(layer, conv_filter_a, conv_filter_b):
    """
    Target function that operates on a Layer
    args:
        layer (Layer): layer input
        conv_filter_a (GeometricFilter): first convolution filter
        conv_filter_b (GeometricFilter): second convolution filter
    """
    # target function that operates on layer but just on the image, so it should be vmapped twice
    vmap_convolve = vmap(geom.convolve, in_axes=(None, 0, None, None, None, None, None, None))
    return vmap_convolve(
        layer.D,
        vmap_convolve(layer.D, layer[0], conv_filter_a, layer.is_torus, None, None, None, None),
        conv_filter_b,
        layer.is_torus,
        None,
        None,
        None,
        None,
    )

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
X = geom.BatchLayer({ 0: random.normal(subkey, shape=(num_images, 1, N, N))}, D, True)
batch_target = vmap(target_function, in_axes=(0, None, None))
Y = geom.BatchLayer({ 0: batch_target(X, conv_filters[1].data, conv_filters[2].data) }, D, True)

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + math.comb(len(conv_filters), 2),))

batch_map_loss = vmap(partial(map_and_loss, conv_filters=conv_filters), in_axes=(None, 0, 0, None, None))
params, _, _ = ml.train(
    X,
    Y,
    lambda params, x, y, key, train: jnp.mean(batch_map_loss(params, x, y, key, train)),
    params,
    key,
    ml.EpochStop(500),
    batch_size=num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(params)

