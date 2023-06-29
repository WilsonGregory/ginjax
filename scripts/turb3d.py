#generate gravitational field
import numpy as np
import sys
from functools import partial
import argparse
import time

import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, layer, key, train, conv_filters, return_params=False):
    target_k = 2
    depth = 1
    max_k = 3 # this is tough

    batch_linear_combination = vmap(geom.linear_combination, in_axes=(0, None))

    param_idx = 0
    for _ in range(2):
        layer, param_idx = ml.batch_conv_layer(
            params,
            int(param_idx),
            layer,
            { 'type': 'fixed', 'filters': conv_filters }, 
            depth, 
            max_k=max_k,
        )
        layer = ml.batch_leaky_relu_layer(layer)
        if (return_params):
            print(layer)

    layer, param_idx = ml.batch_conv_layer(
        params,
        int(param_idx),
        layer,
        { 'type': 'fixed', 'filters': conv_filters }, 
        depth,
        target_k=target_k,
        max_k=max_k, 
    )
    if (return_params):
        print(layer)

    image_block = ml.batch_all_contractions(target_k, layer)
    if (return_params):
        print(image_block.shape)

    net_output = batch_linear_combination(image_block, params[param_idx:(param_idx + image_block.shape[1])])
    param_idx += image_block.shape[1]

    return (net_output, param_idx) if return_params else net_output

# @partial(jit, static_argnums=4)
def map_and_loss(params, layer_x, layer_y, key, train, conv_filters):
    learned_x = net(params, layer_x, key, train, conv_filters)
    return ml.rmse_loss(learned_x, layer_y[2])

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--epochs', help='number of epochs', type=float, default=10)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='folder name to save the results', type=str, default=None)
    parser.add_argument('-l', '--load', help='folder name to load results from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
epochs, lr, batch_size, seed, save_folder, load_folder, verbose = handleArgs(sys.argv)

D = 3
is_torus = False

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
operators = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[1,2], parities=[0], D=D, operators=operators)

# Get Training data
data = np.load('../data/3d_turb/downsampled_1024_to_64.npz')

rho = jnp.array(data['rho'])
vel = jnp.stack([data['vel1'], data['vel2'], data['vel3']], axis=-1)
raw_tau = jnp.stack(
    [
        data['T11'], data['T12'], data['T13'], 
        data['T12'], data['T22'], data['T23'], 
        data['T13'], data['T23'], data['T33'],
    ],
    axis=-1,
)
tau = raw_tau.reshape(raw_tau.shape[:3] + (D,D))

layer_x = geom.BatchLayer(
    { 0: jnp.expand_dims(rho, axis=(0,1)), 1: jnp.expand_dims(vel, axis=(0,1)) }, 
    D, 
    is_torus,
)
layer_y = geom.BatchLayer({ 2: jnp.expand_dims(tau, axis=(0,1))}, D, is_torus)


huge_params = jnp.ones(1000000)
_, num_params = net(huge_params, layer_x, None, True, conv_filters, return_params=True)
print(f'Model params: {num_params}')

key, subkey = random.split(key)
params = 0.1*random.normal(subkey, shape=(num_params,))

del data
del raw_tau
del huge_params

key, subkey = random.split(key)

params, _, _ = ml.train(
    layer_x, 
    layer_y, 
    partial(map_and_loss, conv_filters=conv_filters),
    params,
    subkey,
    ml.EpochStop(epochs=epochs, verbose=verbose),
    batch_size=1,
    optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=1, decay_rate=0.999)),
)
