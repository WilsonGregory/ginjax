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

def plot_results(layer_x, layer_y, axs, titles, conv_filters):
    assert len(axs) == len(titles)

    learned_x = geom.GeometricImage(
        batch_net(params, layer_x, conv_filters)[0],
        0,
        layer_x.D,
        layer_x.is_torus,
    )
    x = geom.GeometricImage(next(iter(layer_x.values()))[0,0], 0, layer_x.D, layer_x.is_torus)
    y = geom.GeometricImage(next(iter(layer_y.values()))[0,0], 0, layer_y.D, layer_y.is_torus)
    
    images = [x, y, learned_x, y - learned_x]
    for image, ax, title in zip(images, axs, titles):
        utils.plot_image(image, ax=ax)
        ax.set_title(title, fontsize=24)

def batch_net(params, layer, conv_filters, return_params=False):
    target_k = 1

    batch_conv_layer = vmap(ml.conv_layer, in_axes=((None,)*3 + (0,) + (None,)*6), out_axes=(0,None))
    batch_add_layer = vmap(lambda layer_a, layer_b: layer_a + layer_b) #better way of doing this?
    batch_linear_combination = vmap(geom.linear_combination, in_axes=(0, None))

    layer, param_idx = batch_conv_layer(params, 0, conv_filters, layer, None, None, None, None, None, None)
    out_layer = layer.empty()
    for dilation in range(1,layer.N): #dilations in parallel
        dilation_out_layer, param_idx = batch_conv_layer(
            params, 
            int(param_idx), 
            conv_filters, 
            layer, 
            None,
            None,
            None,
            None,
            None,
            (dilation,)*D, #rhs_dilation
        )
        out_layer = batch_add_layer(out_layer, dilation_out_layer)

    layer = out_layer

    out_layer = layer.empty()
    for dilation in range(1,int(layer.N / 2)): #dilations in parallel
        dilation_out_layer, param_idx = batch_conv_layer(
            params, 
            int(param_idx), 
            conv_filters, 
            layer, 
            target_k,
            None,
            None,
            None,
            None,
            (dilation,)*D, #rhs_dilation
        )
        out_layer = batch_add_layer(out_layer, dilation_out_layer)

    layer = out_layer
    layer = ml.batch_all_contractions(target_k, layer)

    net_output = batch_linear_combination(layer, params[param_idx:(param_idx + layer.shape[1])])
    param_idx += layer.shape[1]
    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y, key, train, conv_filters):
    # Run x through the net, then return its loss with y
    return jnp.mean(vmap(ml.rmse_loss)(batch_net(params, x, conv_filters), y[1]))

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
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1], parities=[0], D=D, operators=group_actions)

one_point = train_X.get_subset(jnp.array([0]))

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(100000)
    _, num_params = batch_net(
        huge_params,
        one_point,
        conv_filters,
        return_params=True,
    )

    print('Num params:', num_params)

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    optimizer = optax.adam(optax.exponential_decay(
        lr, 
        transition_steps=int(train_X.L / batch_size), 
        decay_rate=0.995,
    ))

    params, _, _ = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, conv_filters=conv_filters),
        params,
        key,
        ml.ValLoss(patience=20, verbose=verbose),
        batch_size=batch_size,
        optimizer=optimizer,
        validation_X=validation_X,
        validation_Y=validation_Y,
        save_params=save_file,
    )

    if (save_file):
        jnp.save(save_file, params)

plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = 'STIXGeneral'
plt.tight_layout()

titles = ['Input', 'Ground Truth', 'Prediction', 'Difference']
fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(24,12))
plot_results(train_X.get_subset(jnp.array([0])), train_Y.get_subset(jnp.array([0])), axs[0], titles, conv_filters)
plot_results(test_X.get_subset(jnp.array([0])), test_Y.get_subset(jnp.array([0])), axs[1], titles, conv_filters)
plt.savefig(outfile)

print('Full Test loss:', map_and_loss(params, test_X, test_Y, None, None, conv_filters=conv_filters))
one_test_loss = map_and_loss(
    params, 
    test_X.get_subset(jnp.array([0])), 
    test_Y.get_subset(jnp.array([0])), 
    None,
    None,
    conv_filters=conv_filters,
)
print(f'One Test loss: {one_test_loss}')