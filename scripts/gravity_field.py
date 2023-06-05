#generate gravitational field
import sys
from functools import partial
import argparse
import time
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
from jax import vmap
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils
from geometricconvolutions.data import get_gravity_data as get_data

def plot_results(x, y, axs, titles):
    assert len(axs) == len(titles)

    learned_x = geom.GeometricImage(
        net(params, x.data, x.D, x.is_torus, conv_filters, y.data),
        x.parity,
        x.D,
        x.is_torus,
    )
    images = [x, y, learned_x, y - learned_x]
    for image, ax, title in zip(images, axs, titles):
        utils.plot_image(image, ax=ax)
        ax.set_title(title, fontsize=24)

def net(params, x, D, is_torus, conv_filters, target_img, return_params=False):
    #x is a layer
    # Note that the target image is only used for its shape
    img_N, img_k = geom.parse_shape(x.shape, D)
    _, target_k = geom.parse_shape(target_img.shape, D)
    layer = { img_k: jnp.expand_dims(x, axis=0) }
    conv_filters = ml.make_layer(conv_filters)

    layer, param_idx = ml.conv_layer(params, 0, conv_filters, layer, D, is_torus)
    out_layer = {}
    for dilation in range(1,img_N): #dilations in parallel
        dilation_out_layer, param_idx = ml.conv_layer(
            params, 
            int(param_idx), 
            conv_filters, 
            layer, 
            D, 
            is_torus, 
            rhs_dilation=(dilation,)*D,
        )
        out_layer = ml.merge_layers(out_layer, dilation_out_layer)

    layer = out_layer

    out_layer = {}
    for dilation in range(1,int(img_N / 2)):
        dilation_out_layer, param_idx = ml.conv_layer(
            params, 
            int(param_idx), 
            conv_filters, 
            layer, 
            D, 
            is_torus, 
            target_k=target_k, #final_layer, make sure out output is target_img shaped
            rhs_dilation=(dilation,)*D,
        )
        out_layer = ml.merge_layers(out_layer, dilation_out_layer)

    layer = out_layer
    layer = ml.all_contractions(target_k, layer, D)

    net_output = geom.linear_combination(layer, params[param_idx:(param_idx + len(layer))])
    param_idx += len(layer)
    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y, conv_filters, D, is_torus):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, D, is_torus, conv_filters, y), y)

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', help='where to save the image', type=str)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.outfile,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
outfile, lr, batch_size, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2
num_points = 5
num_train_images = 5
num_test_images = 10
num_val_images = 5

key = random.PRNGKey(seed if seed else time.time_ns())

key, subkey = random.split(key)
validation_X, validation_Y = get_data(N, D, num_points, subkey, num_val_images)

key, subkey = random.split(key)
test_X, test_Y = get_data(N, D, num_points, subkey, num_test_images)

key, subkey = random.split(key)
train_X, train_Y = get_data(N, D, num_points, subkey, num_train_images)

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

one_point = train_X[0]

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(100000)
    _, num_params = net(
        huge_params,
        one_point.data,
        one_point.D,
        one_point.is_torus,
        conv_filters,
        train_Y[0].data,
        return_params=True,
    )

    print('Num params:', num_params)

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    params, _, _ = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, D=one_point.D, is_torus=one_point.is_torus, conv_filters=conv_filters),
        params,
        key,
        ml.ValLoss(patience=20, verbose=verbose),
        batch_size=batch_size,
        optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=int(len(train_X) / batch_size), decay_rate=0.995)),
        validation_X=validation_X,
        validation_Y=validation_Y,
        save_params=save_file,
    )

    if (save_file):
        jnp.save(save_file, params)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'
plt.tight_layout()

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,12))
plot_results(train_X[0], train_Y[0], axs[0], ['Input', 'Ground Truth', 'Prediction', 'Difference'])
plot_results(test_X[0], test_Y[0], axs[1], ['Input', 'Ground Truth', 'Prediction', 'Difference'])
plt.savefig(outfile)

vmap_map_loss = vmap(
    partial(map_and_loss, conv_filters=conv_filters, D=test_X[0].D, is_torus=test_X[0].is_torus),
    in_axes=(None, 0, 0),
)
test_loss = jnp.mean(vmap_map_loss(
    params, 
    geom.BatchGeometricImage.from_images(test_X).data, 
    geom.BatchGeometricImage.from_images(test_Y).data, 
))
print('Full Test loss:', test_loss)
print(f'One Test loss: {map_and_loss(params, test_X[0].data, test_Y[0].data, conv_filters, test_X[0].D, test_X[0].is_torus)}')