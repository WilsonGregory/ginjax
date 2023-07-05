#generate gravitational field
import sys
import numpy as np
from functools import partial
import argparse
import time
import math
import matplotlib.pyplot as plt
from collections import defaultdict

import jax.numpy as jnp
import jax.random as random
from jax import vmap
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
from geometricconvolutions.data import get_gravity_data as get_data

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

    batch_conv_layer = vmap(ml.conv_layer, in_axes=((None,)*2 + (0,) + (None,)*7), out_axes=(0,None))
    batch_add_layer = vmap(lambda layer_a, layer_b: layer_a + layer_b) #better way of doing this?

    layer, params = batch_conv_layer(params, conv_filters, layer, None, None, return_params, None, None, None, None)
    out_layer = layer.empty()
    for dilation in range(1,layer.N): #dilations in parallel
        dilation_out_layer, params = batch_conv_layer(
            params, 
            conv_filters, 
            layer, 
            None,
            None,
            return_params, #mold_params
            None,
            None,
            None,
            (dilation,)*D, #rhs_dilation
        )
        out_layer = batch_add_layer(out_layer, dilation_out_layer)

    layer = out_layer

    out_layer = layer.empty()
    for dilation in range(1,int(layer.N / 2)): #dilations in parallel
        dilation_out_layer, params = batch_conv_layer(
            params, 
            conv_filters, 
            layer, 
            target_k,
            None,
            return_params, #mold_params
            None,
            None,
            None,
            (dilation,)*D, #rhs_dilation
        )
        out_layer = batch_add_layer(out_layer, dilation_out_layer)

    layer = out_layer
    layer = ml.batch_all_contractions(target_k, layer)
    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)
    return (layer, params) if return_params else layer

def map_and_loss(params, x, y, key, train, conv_filters):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.l2_loss)(batch_net(params, x, key, train, conv_filters)[1], y[1]))

def baseline_net(params, layer, key, train, return_params=False):
    M = 3
    out_channels = 2

    batch_add_layer = vmap(lambda layer_a, layer_b: layer_a + layer_b)

    layer, params = ml.batch_conv_layer(
        params,
        layer, 
        { 'type': 'free', 'M': M, 'filter_k_set': (0,) },
        depth=out_channels,
        mold_params=return_params,
    )

    out_layer = layer.empty()
    for dilation in range(1,layer.N): #dilations in parallel
        dilation_out_layer, params = ml.batch_conv_layer(
            params, 
            layer,
            { 'type': 'free', 'M': M, 'filter_k_set': (0,) },
            depth=out_channels,
            mold_params=return_params,
            rhs_dilation=(dilation,)*D,
        )
        out_layer = batch_add_layer(out_layer, dilation_out_layer)

    layer = out_layer

    out_layer = layer.empty()
    for dilation in range(1,int(layer.N / 2)): #dilations in parallel
        dilation_out_layer, params = ml.batch_conv_layer(
            params, 
            layer,
            { 'type': 'free', 'M': M, 'filter_k_set': (0,) },
            depth=D, #out depth is D because we turn it into k=1
            mold_params=return_params,
            rhs_dilation=(dilation,)*D,
        )
        # turn the out channels into the vector field k=1
        dilation_out_image = dilation_out_layer[0].transpose(0,2,3,1) # move channel to end
        reshaped_layer = geom.BatchLayer(
            { 1: jnp.expand_dims(dilation_out_image, axis=1) },
            dilation_out_layer.D,
            dilation_out_layer.is_torus,
        )
        out_layer = batch_add_layer(out_layer, reshaped_layer)

    layer, params = ml.batch_channel_collapse(params, out_layer, mold_params=return_params)
    return (layer, params) if return_params else layer

def baseline_map_and_loss(params, x, y, key, train):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.l2_loss)(baseline_net(params, x, key, train)[1], y[1]))

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_times', help='number of runs in the benchmark', type=int, default=5)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
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
num_points = 5
num_train_images_range = [5,10,20,50]
num_test_images = 10
num_val_images = 5

key = random.PRNGKey(seed if seed else time.time_ns())

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1], parities=[0], D=D, operators=group_actions)

key, subkey = random.split(key)
one_point_x, _ = get_data(N, D, num_points, subkey, 1)
one_point_x = one_point_x

models = [
    (
        'GI-Net',
        partial(map_and_loss, conv_filters=conv_filters), 
        partial(batch_net, conv_filters=conv_filters), #get params tree
        0.005,
    ),
    (
        'Baseline',
        baseline_map_and_loss, 
        baseline_net,
        0.005,
    ),
]

models_init = []
for model_name, model_map_and_loss, model, lr in models:
    params_tree = model(defaultdict(lambda: None), one_point_x, None, True, return_params=True)[1]
    print(f'{model_name}: {ml.count_params(params_tree)} params')
    models_init.append((model_name, model_map_and_loss, params_tree, lr))

if (not load_file):
    all_train_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_val_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_test_loss = np.zeros((num_times, len(num_train_images_range), len(models)))
    all_time_elapsed = np.zeros((num_times, len(num_train_images_range), len(models)))
    for i in range(num_times):
        key, subkey = random.split(key)
        validation_X, validation_Y = get_data(N, D, num_points, subkey, num_val_images)

        key, subkey = random.split(key)
        test_X, test_Y = get_data(N, D, num_points, subkey, num_test_images)

        for j, num_train_images in enumerate(num_train_images_range):
            batch_size = math.ceil(num_train_images / 5)

            key, subkey = random.split(key)
            train_X, train_Y = get_data(N, D, num_points, subkey, num_train_images)

            for k, (model_name, model, params_tree, lr) in enumerate(models_init):
                print(f'Iter {i}, train size {num_train_images}, model {model_name}')

                key, subkey = random.split(key)
                params = ml.recursive_init_params(
                    params_tree, 
                    subkey,
                    initializers={ 
                        ml.CHANNEL_COLLAPSE: channel_collapse_init, 
                        ml.CONV: conv_init,
                        ml.CONV_OLD: ml.conv_old_init,
                    },
                )
                key, subkey = random.split(key)

                start_time = time.time()
                params, train_loss, val_loss = ml.train(
                    train_X,
                    train_Y,
                    model,
                    params,
                    subkey,
                    ml.ValLoss(patience=20, verbose=verbose),
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
plt.title('Gravitational Field Loss vs. Number of Training Points')
plt.xlabel('Number of Training Points')
plt.ylabel('RMSE Loss')
plt.savefig(f'../images/gravity/gravity_loss_chart_{seed}.png')