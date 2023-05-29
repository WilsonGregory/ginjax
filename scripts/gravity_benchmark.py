#generate gravitational field
import sys
import numpy as np
import itertools as it
from functools import partial
import argparse
import time
import math
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
from jax import vmap, jit
import jax.lax
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
from geometricconvolutions.data import get_gravity_data as get_data

def net(params, x, D, is_torus, conv_filters, target_img, return_params=False):
    #x is a layer
    # Note that the target image is only used for its shape
    img_N, img_k = geom.parse_shape(x.shape, D)
    _, target_k = geom.parse_shape(target_img.shape, D)
    layer = { img_k: jnp.expand_dims(x, axis=0) }
    conv_filters = ml.make_layer(conv_filters)

    layer, param_idx = ml.conv_layer(params, 0, conv_filters, layer, D, is_torus)
    layer, param_idx = ml.conv_layer(
        params, 
        int(param_idx), 
        conv_filters, 
        layer, 
        D, 
        is_torus, 
        dilations=tuple(range(1,img_N)),
    )

    layer, param_idx = ml.conv_layer(
        params, 
        int(param_idx), 
        conv_filters, 
        layer, 
        D,
        is_torus,
        target_k, #final_layer, make sure out output is target_img shaped
        dilations=tuple(range(1,int(img_N / 2))),
    )
    layer = ml.all_contractions(target_k, layer, D)

    net_output = geom.linear_combination(layer, params[param_idx:(param_idx + len(layer))])
    param_idx += len(layer)
    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y, conv_filters, D, is_torus):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, D, is_torus, conv_filters, y), y)

@partial(jit, static_argnums=[1,2,3,4,6,7,8])
def baseline_conv_layer(params, param_idx, D, M, img_N, layer, out_channels, dilations, dilations_to_channels=True):
    collected = []
    filter_shape = (M,)*D + (layer.shape[-1],out_channels)
    filter_size = np.multiply.reduce(filter_shape)
    for dilation in dilations:
        filter_formatted = params[param_idx:param_idx + filter_size].reshape(filter_shape)
        param_idx += filter_size

        rhs_dilation = (dilation,)*D
        padding_literal = ((dilation,) * 2 for dilation in rhs_dilation)

        res = jax.lax.conv_general_dilated(
            layer, #lhs
            filter_formatted, #rhs
            (1,)*D, #stride
            padding_literal,
            rhs_dilation=rhs_dilation,
            dimension_numbers=('NHWC','HWIO','NHWC'),
        )
        collected.append(res)

    if dilations_to_channels:
        return (
            jnp.stack(collected).transpose(1,2,3,4,0).reshape((1,) + (img_N,)*D + (len(collected)*out_channels,)),
            param_idx,
        )
    else:
        return jnp.stack(collected), param_idx

def baseline_net(params, x, D, is_torus, target_img, return_params=False):
    img_N, _ = geom.parse_shape(x.shape, D)
    M = 3
    out_channels = 2

    layer = x.reshape((1,) + x.shape + (1,))

    layer, param_idx = baseline_conv_layer(params, 0, D, M, img_N, layer, out_channels, (1,))
    layer, param_idx = baseline_conv_layer(params, int(param_idx), D, M, img_N, layer, out_channels, range(1, img_N))
    layer, param_idx = baseline_conv_layer(
        params, 
        int(param_idx), 
        D, 
        M, 
        img_N, 
        layer, 
        out_channels, 
        range(1,int(img_N / 2)),
        dilations_to_channels=False,
    )

    net_output = geom.linear_combination(layer, params[param_idx:(param_idx + len(layer))])
    param_idx += len(layer)
    return (net_output, param_idx) if return_params else net_output

def baseline_map_and_loss(params, x, y, D, is_torus):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(baseline_net(params, x, D, is_torus, y), y)

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
seed, save_file, load_file, verbose = handleArgs(sys.argv)

num_times = 5
N = 16
D = 2
num_points = 5
num_train_images_range = [5,10,20,50]
num_test_images = 10
num_val_images = 5

key = random.PRNGKey(seed if seed else time.time_ns())

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0,1],
    parities=[0],
    D=D,
    operators=group_actions,
    return_list=True,
)

key, subkey = random.split(key)
one_point_x, one_point_y = get_data(N, D, num_points, subkey, 1)
one_point_x = one_point_x[0]
one_point_y = one_point_y[0]

huge_params = jnp.ones(100000)

_, num_params = net(
    huge_params,
    one_point_x.data,
    one_point_x.D,
    one_point_x.is_torus,
    conv_filters,
    one_point_y.data,
    return_params=True,
)
print('Model Params:', num_params)

_, baseline_num_params = baseline_net(
    huge_params,
    one_point_x.data,
    one_point_x.D,
    one_point_x.is_torus,
    one_point_y.data,
    return_params=True,
)
print('Baseline Params:', baseline_num_params)
models = [
    (
        partial(map_and_loss, D=one_point_x.D, is_torus=one_point_x.is_torus, conv_filters=conv_filters), 
        num_params,
        0.005,
    ),
    (
        partial(baseline_map_and_loss, D=one_point_x.D, is_torus=one_point_x.is_torus), 
        baseline_num_params,
        0.005,
    ),
]

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

            for k, (model, num_params, lr) in enumerate(models):
                print(f'Iter {i}, train size {num_train_images}, model {k}')
                key, subkey = random.split(key)
                params = 0.1*random.normal(subkey, shape=(num_params,))

                start_time = time.time()
                params, train_loss, val_loss = ml.train_early_stopping(
                    train_X,
                    train_Y,
                    model,
                    params,
                    key,
                    batch_size=batch_size,
                    optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=int(num_train_images / batch_size), decay_rate=0.995)),
                    validation_X=validation_X,
                    validation_Y=validation_Y,
                    patience=20,
                    verbose=verbose,
                )
                all_time_elapsed[i,j,k] = time.time() - start_time
                all_train_loss[i,j,k] = train_loss[-1]
                all_val_loss[i,j,k] = val_loss[-1]

                vmap_map_loss = vmap(model, in_axes=(None, 0, 0))
                test_loss = jnp.mean(vmap_map_loss(
                    params, 
                    geom.BatchGeometricImage.from_images(test_X).data, 
                    geom.BatchGeometricImage.from_images(test_Y).data, 
                ))
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