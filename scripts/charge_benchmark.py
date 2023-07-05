import sys
import numpy as np
from functools import partial
import argparse
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict

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
    for k, params_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        out_params[k] = 0.1*random.normal(subkey, params_block.shape)

    return out_params

def conv_init(rand_key, tree):
    params = {}
    for k, d in tree[ml.CONV_FREE].items():
        params[k] = {}
        for filter_k, filter_block in d.items():
            rand_key, subkey = random.split(rand_key)
            params[k][filter_k] = 0.1*random.normal(subkey, shape=filter_block.shape)

    return { ml.CONV_FREE: params }

def batch_net(params, layer, key, train, conv_filters, return_params=False):
    target_k = 1
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
        target_k, #final_layer, make sure out output is target_img shaped
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

def map_and_loss(params, x, y, key, train, conv_filters):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.l2_loss)(batch_net(params, x, key, train, conv_filters)[1], y[1]))

def baseline_net(params, layer, key, train, return_params=False):
    M = 3
    out_channels = 20
    #move the vector to the channel slot
    transposed_block = layer[1].transpose(0,1,4,2,3)
    reshaped_block = transposed_block.reshape((transposed_block.shape[0],) + transposed_block.shape[2:])
    layer = geom.BatchLayer({ 0: reshaped_block }, layer.D, layer.is_torus)

    for dilation in [1,2,4,2,1,1,2,1]:
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            { 'type': 'free', 'M': M, 'filter_k_set': (0,) },
            depth=out_channels, 
            mold_params=return_params,
            rhs_dilation=(dilation,)*D,
        )
        layer = ml.batch_leaky_relu_layer(layer)

    layer, params = ml.batch_conv_layer(
        params, 
        layer, 
        { 'type': 'free', 'M': M, 'filter_k_set': (0,) }, 
        depth=2,
        mold_params=return_params,
    )
    layer = geom.BatchLayer({ 1: jnp.expand_dims(jnp.moveaxis(layer[0], 1, -1), 1) }, layer.D, layer.is_torus)

    return (layer, params) if return_params else layer

def baseline_map_and_loss(params, x, y, key, train):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.l2_loss)(baseline_net(params, x, key, train)[1], y[1]))

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_times', help='number of runs in the benchmark', type=int, default=5)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save loss values', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load loss values', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.num_times,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
num_times, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2
num_steps = 10 #number of steps to do
warmup_steps = 1 #number of initial steps to run. This ensures that particles don't start directly adjacent.
delta_t = 1 #distance taken in a single step
s = 0.2 # used for transforming the input/output data

num_points = 3
# num_train_images_range = [5,10,20,50,100]
num_train_images_range = [5]
num_test_images = 10
num_val_images = 10

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[1,2], parities=[0], D=D, operators=group_actions)

key, subkey = random.split(key)
one_point_x, _ = get_data(N, D, num_points, num_steps, delta_t, s, subkey, 1, warmup_steps=warmup_steps)

models = [
    (
        'GI-Net',
        partial(map_and_loss, conv_filters=conv_filters), 
        partial(batch_net, conv_filters=conv_filters),
        0.005,
    ),
    (
        'Baseline',
        partial(baseline_map_and_loss), 
        baseline_net,
        0.001,
    ),
]

models_init = []
for model_name, model_map_and_loss, model, lr in models:
    key, subkey = random.split(key)
    get_params = ml.init_params(
        model, 
        one_point_x, 
        subkey, 
        return_func=True, 
        override_initializers={
            ml.CHANNEL_COLLAPSE: channel_collapse_init, 
            ml.CONV: conv_init,
        },
    )
    # params_tree = model(defaultdict(lambda: None), one_point_x, None, True, return_params=True)[1]
    print(f'{model_name}: {ml.count_params(get_params(subkey))} params')
    models_init.append((model_name, model_map_and_loss, get_params, lr))

if (not load_file):
    all_train_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_val_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_test_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_time_elapsed = np.zeros((num_times, len(num_train_images_range), len(models)))
    for i in range(num_times):

        key, subkey = random.split(key)
        validation_X, validation_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_val_images, warmup_steps=warmup_steps)

        key, subkey = random.split(key)
        test_X, test_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_test_images, warmup_steps=warmup_steps)

        for j, num_train_images in enumerate(num_train_images_range):
            batch_size = math.ceil(num_train_images / 5)

            key, subkey = random.split(key)
            train_X, train_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_train_images, warmup_steps=warmup_steps)

            for k, (model_name, model, get_params, lr) in enumerate(models_init):
                print(f'Iter {i}, train size {num_train_images}, model {model_name}')
                key, subkey = random.split(key)
                params = get_params(subkey)
                # ml.recursive_init_params(
                #     params_tree, 
                #     subkey,
                #     initializers={ 
                #         ml.CHANNEL_COLLAPSE: channel_collapse_init, 
                #         ml.CONV: conv_init,
                #         ml.CONV_OLD: ml.conv_old_init,
                #     },
                # )
                key, subkey = random.split(key)

                start_time = time.time()
                params, train_loss, val_loss = ml.train(
                    train_X,
                    train_Y,
                    model,
                    params,
                    subkey,
                    ml.ValLoss(patience=5, verbose=verbose),
                    batch_size=batch_size,
                    optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=int(num_train_images / batch_size), decay_rate=0.995)),
                    validation_X=validation_X,
                    validation_Y=validation_Y,
                )
                all_time_elapsed[i,j,k] = time.time() - start_time
                all_train_loss[i,j,k] = train_loss[-1]
                all_val_loss[i,j,k] = val_loss[-1]

                test_loss = model(params, test_X, test_Y, None, None)
                all_test_loss[i,j,k] = test_loss
                print('Full Test loss:', test_loss)

        if (save_file):
            np.save(
                save_file, 
                {
                    'all_train_loss': all_train_loss, 
                    'all_val_loss': all_val_loss, 
                    'all_test_loss': all_test_loss, 
                    'all_time_elapsed': all_time_elapsed,
                },
            )
else:
    data = np.load(load_file, allow_pickle=True).item()
    all_train_loss = data['all_train_loss']
    all_val_loss = data['all_val_loss']
    all_test_loss = data['all_test_loss']
    all_time_elapsed = data['all_time_elapsed']

model_train_loss = np.mean(all_train_loss[:,:,0], axis=0)
model_val_loss = np.mean(all_val_loss[:,:,0], axis=0)
model_test_loss = np.mean(all_test_loss[:,:,0], axis=0)

baseline_train_loss = np.mean(all_train_loss[:,:,1], axis=0)
baseline_val_loss = np.mean(all_val_loss[:,:,1], axis=0)
baseline_test_loss = np.mean(all_test_loss[:,:,1], axis=0)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'
plt.tight_layout()

plt.plot(num_train_images_range, model_train_loss, label='GI-Net train', color='b', marker='o', linestyle='dashed')
plt.plot(num_train_images_range, model_test_loss, label='GI-Net test', color='b', marker='o')
plt.plot(num_train_images_range, baseline_train_loss, label='Baseline train', color='r', marker='o', linestyle='dashed')
plt.plot(num_train_images_range, baseline_test_loss, label='Baseline test', color='r', marker='o')
plt.legend()
plt.title('Charge Field Loss vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('RMSE Loss')
plt.savefig(f'../images/charge/charge_loss_chart_{seed}.png')
