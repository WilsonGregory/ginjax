import jax.numpy as jnp
from jax import random, vmap
import time
import itertools as it
import math
import optax
from functools import partial

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def batch_net(params, layer, conv_filters):
    """
    The neural network the works on a BatchLayer. It will convolve with all combinations of the
    conv_filters, then take a parameterized linear combination and return that batch image block.
    args:
        params (jnp.array): the parameters of the network
        layer (BatchLayer): the input layer, will be a dict { 0: (batch, 1, N, N) } of that shape
        conv_filters (Layer): the conv_filters as a layer
    """
    channel_convolve = vmap(geom.convolve, in_axes=(None, 0, None, None, None, None, None, None))
    batch_convolve = vmap(channel_convolve, in_axes=(None, 0, None, None, None, None, None, None))
    batch_linear_combination = vmap(geom.linear_combination, in_axes=(0, None))

    out_image_block = None

    for i,j in it.combinations_with_replacement(range(len(conv_filters[(0,0)])), 2):
        filter_a = conv_filters[(0,0)][i]
        filter_b = conv_filters[(0,0)][j]
        convolved_image = batch_convolve(layer.D, layer[(0,0)], filter_a, layer.is_torus, None, None, None, None)
        res_image = batch_convolve(layer.D, convolved_image, filter_b, layer.is_torus, None, None, None, None)

        if (out_image_block is None):
            out_image_block = res_image
        else:
            out_image_block = jnp.concatenate((out_image_block, res_image), axis=1)

    return batch_linear_combination(out_image_block, params)

def map_and_loss(params, x, y, key, train, conv_filters):
    """
    Given an input batch layer x and a target batch layer y, apply the neural network to the input
    layer and then calculate the root mean squared error loss with the target batch layer y. The first
    5 arguments are what is expected in the ml.train function, but we don't use key and train.
    args:
        params (jnp.array) array the learned parameters for the model
        x (BatchLayer): the input data to the network
        y (BatchLayer): the target data for the network
        key (rand key): Jax random key if the network needs a source of randomness. Not used here.
        train (bool): whether the network is being called during training, not used here.
        conv_filters (Layer): the convolution filters as a Layer, the version of the function passed
            to ml.train will already have this set using functools.partial
    """
    return jnp.mean(vmap(ml.rmse_loss)(batch_net(params, x, conv_filters), y[(0,0)]))

def target_function(image, conv_filter_a, conv_filter_b):
    """
    Target function that applies two convolutions in sequence
    args:
        image (GeometricImage): layer input
        conv_filter_a (GeometricFilter): first convolution filter
        conv_filter_b (GeometricFilter): second convolution filter
    """
    return image.convolve_with(conv_filter_a).convolve_with(conv_filter_b)

#Main
key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
M = 3  #filter image size
num_images = 10

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_unique_invariant_filters(M=M, k=0, parity=0, D=D, operators=group_actions)

key, subkey = random.split(key)
X_images = [geom.GeometricImage(data, 0, D, True) for data in random.normal(subkey, shape=(num_images, N, N))]
Y_images = [target_function(image, conv_filters[1], conv_filters[2]) for image in X_images]

key, subkey = random.split(key)
params = random.normal(subkey, shape=(len(conv_filters) + math.comb(len(conv_filters), 2),))

filter_layer = geom.Layer.from_images(conv_filters)
X_layer = geom.BatchLayer.from_images(X_images)
Y_layer = geom.BatchLayer.from_images(Y_images)

params, _, _ = ml.train(
    X_layer,
    Y_layer,
    partial(map_and_loss, conv_filters=filter_layer),
    params,
    key,
    ml.EpochStop(500, verbose=1),
    batch_size=num_images,
    optimizer=optax.adam(optax.exponential_decay(0.1, transition_steps=1, decay_rate=0.99)),
)

print(params)

