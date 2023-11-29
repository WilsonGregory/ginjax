import sys
import numpy as np
from functools import partial
import argparse
import time
import math
import matplotlib.pyplot as plt

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
from geometricconvolutions.data import get_charge_data as get_data

def channel_collapse_init(rand_key, tree):
    # Use old guassian normal initialization instead of Kaiming
    out_params = {}
    for key, params_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        out_params[key] = 0.1*random.normal(subkey, params_block.shape)

    return out_params

def conv_init(rand_key, tree):
    params = {}
    for key, d in tree[ml.CONV_FREE].items():
        params[key] = {}
        for filter_key, filter_block in d.items():
            rand_key, subkey = random.split(rand_key)
            params[key][filter_key] = 0.1*random.normal(subkey, shape=filter_block.shape)

    return { ml.CONV_FREE: params }

def batch_net_old(params, layer, key, train, conv_filters, return_params=False):
    target_k = 1
    target_parity = 0
    max_k = 5

    batch_conv_layer = vmap(ml.conv_layer, in_axes=((None,)*2 + (0,) + (None,)*7), out_axes=(0,None))

    for dilation in [1,2,4,2,1,1,2,1]: #dilated layers in sequence
        layer, params = batch_conv_layer(
            params,
            conv_filters,
            layer,
            None,
            max_k,
            return_params, #mold_params
            None, 
            None,
            None, 
            (dilation,)*layer.D, # rhs_dilation
        )
        layer = ml.batch_leaky_relu_layer(layer)

    layer, params = batch_conv_layer(
        params, 
        conv_filters,
        layer,
        (target_k,target_parity), #final_layer, make sure out output is target_img shaped
        None,
        return_params, #mold_params
        None,
        None,
        None,
        None,
    )
    layer = ml.batch_all_contractions(target_k, layer)
    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer

def batch_net(params, layer, key, train, conv_filters, return_params=False):
    depth = 5

    for dilation in [1,2,4,2,1,1,2,1]: #dilated layers in sequence
        layer, params = ml.batch_conv_contract(
            params, 
            layer, 
            conv_filters, 
            depth, 
            ((0,0),(0,1),(1,0),(1,1)),
            mold_params=return_params,
            rhs_dilation=(dilation,)*layer.D,
        )
        layer = ml.scalar_activation(layer, jnp.tanh)

    layer, params = ml.batch_conv_contract(
        params, 
        layer, 
        conv_filters, 
        depth, 
        ((1,0),),
        mold_params=return_params,
    )

    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer

def map_and_loss(params, x, y, key, train, net):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.l2_loss)(net(params, x, key, train)[(1,0)], y[(1,0)]))

def baseline_net(params, layer, key, train, return_params=False):
    M = 3
    out_channels = 20
    #move the vector to the channel slot
    transposed_block = layer[(1,0)].transpose(0,1,4,2,3)
    reshaped_block = transposed_block.reshape((transposed_block.shape[0],) + transposed_block.shape[2:])
    layer = geom.BatchLayer({ (0,0): reshaped_block }, layer.D, layer.is_torus)

    for dilation in [1,2,4,2,1,1,2,1]:
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            { 'type': 'free', 'M': M, 'filter_key_set': { (0,0) } },
            depth=out_channels, 
            mold_params=return_params,
            rhs_dilation=(dilation,)*D,
        )
        layer = ml.batch_leaky_relu_layer(layer)

    layer, params = ml.batch_conv_layer(
        params, 
        layer, 
        { 'type': 'free', 'M': M, 'filter_key_set': { (0,0) } }, 
        depth=2,
        mold_params=return_params,
    )
    layer = geom.BatchLayer({ (1,0): jnp.expand_dims(jnp.moveaxis(layer[(0,0)], 1, -1), 1) }, layer.D, layer.is_torus)

    return (layer, params) if return_params else layer

def get_all_data(num_train_images, rand_key):
    """
    Function that generates train, val, and test data. Benchmarked value is the number of training points.
    """
    num_steps = 10 #number of steps to do
    warmup_steps = 1 #number of initial steps to run. This ensures that particles don't start directly adjacent.
    delta_t = 1 #distance taken in a single step
    s = 0.2 # used for transforming the input/output data
    num_points = 3
    num_val_images = 10
    num_test_images = 10

    rand_key, subkey = random.split(rand_key)
    train_X, train_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_train_images, warmup_steps=warmup_steps)

    rand_key, subkey = random.split(rand_key)
    validation_X, validation_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_val_images, warmup_steps=warmup_steps)

    rand_key, subkey = random.split(rand_key)
    test_X, test_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_test_images, warmup_steps=warmup_steps)
    
    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y

def train_and_eval(data, rand_key, net, lr, override_initializers={}):
    X_train, Y_train, X_val, Y_val, X_test, Y_test = data 
    batch_size = math.ceil(X_train.L / 5)

    rand_key, subkey = random.split(rand_key)
    init_params = ml.init_params(
        net,
        X_train.get_subset(jnp.array([0])), 
        subkey, 
        override_initializers=override_initializers,
    )

    rand_key, subkey = random.split(rand_key)
    start_time = time.time()
    params, train_loss, val_loss = ml.train(
        X_train,
        Y_train,
        partial(map_and_loss, net=net),
        init_params,
        subkey,
        ml.ValLoss(patience=20, verbose=verbose),
        batch_size=batch_size,
        optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=int(X_train.L / batch_size), decay_rate=0.995)),
        validation_X=X_val,
        validation_Y=Y_val,
    )

    rand_key, subkey = random.split(rand_key)
    test_loss = partial(map_and_loss, net=net)(params, X_test, Y_test, subkey, False)
    return train_loss[-1], val_loss[-1], test_loss, time.time() - start_time

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_trials', help='number of runs in the benchmark', type=int, default=5)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save loss values', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load loss values', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.num_trials,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
num_trials, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2

num_train_images_range = [5,10,20,50,100]

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2,3,4], parities=[0,1], D=D, operators=group_actions)
conv_filters_old = geom.get_invariant_filters(Ms=[3], ks=[1,2], parities=[0], D=D, operators=group_actions)

models = [
    (
        'GI-Net',
        partial(
            train_and_eval,
            net=partial(batch_net, conv_filters=conv_filters),
            lr=0.005,
        ),
    ),
    (
        'GI-Net Old',
        partial(
            train_and_eval,
            net=partial(batch_net_old, conv_filters=conv_filters_old),
            lr=0.005,
            override_initializers={
                ml.CHANNEL_COLLAPSE: channel_collapse_init, 
                ml.CONV: conv_init,
            },
        ),
    ),
    (
        'Baseline',
        partial(
            train_and_eval,
            net=partial(baseline_net),
            lr=0.001,
        ),
    ),
]

if (not load_file):
    key, subkey = random.split(key)
    results = ml.benchmark(
        get_all_data,
        models,
        subkey,
        'Num points',
        num_train_images_range, 
        num_trials=num_trials,
        num_results=4,
    )

    if save_file:
        np.save(save_file, results)
else:
    results = np.load(load_file, allow_pickle=True).item()

all_train_loss = results[:,:,:,0]
all_val_loss = results[:,:,:,1]
all_test_loss = results[:,:,:,2]
all_time_elapsed = results[:,:,:,3]

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'
plt.tight_layout()

colors = ['b', 'g', 'r']
for k, (model_name, model) in enumerate(models):
    plt.plot(
        num_train_images_range, 
        np.mean(all_train_loss[:,:,k], axis=0), 
        label=f'{model_name} Train', 
        color=colors[k],
        marker='o', 
        linestyle='dashed',
    )

    plt.plot(
        num_train_images_range, 
        np.mean(all_test_loss[:,:,k], axis=0), 
        label=f'{model_name} Test', 
        color=colors[k],
        marker='o', 
    )

plt.legend()
plt.title('Charge Field Loss vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('RMSE Loss')
plt.savefig(f'../images/charge/charge_loss_chart_{seed}.png')
