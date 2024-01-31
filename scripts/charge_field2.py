#generate gravitational field
import sys
from functools import partial
import argparse
import time
import matplotlib.pyplot as plt

import jax
import jax.numpy as jnp
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils
from geometricconvolutions.data import get_charge_data as get_data

def plot_results(layer_x, layer_y, axs, titles, conv_filters):
    assert len(axs) == len(titles)

    learned_x = geom.GeometricImage(
        batch_net(params, layer_x, None, False, conv_filters)[(1,0)][0,0],
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

@partial(jax.jit, static_argnums=4)
def map_and_loss(params, x, y, key, train, conv_filters):
    # Run x through the net, then return its loss with y
    return jnp.mean(jax.vmap(ml.l2_loss)(batch_net(params, x, key, train, conv_filters)[(1,0)], y[(1,0)]))

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', help='where to save the image', type=str)
    parser.add_argument('--data_gif', help='where the save the gif of the charges moving', type=str)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.01)
    parser.add_argument('-d', '--decay', help='decay rate of learning rate', type=float, default=0.995)
    parser.add_argument('-batch', help='batch size', type=int, default=10)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.outfile,
        args.data_gif,
        args.lr,
        args.decay,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
outfile, data_gif, lr, decay, batch_size, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2
num_steps = 10 #number of steps to do
warmup_steps = 1 #number of initial steps to run, ensures that particles don't start directly adjacent.
delta_t = 1 #distance taken in a single step
s = 0.2 # used for transforming the input/output data

num_points = 3
num_train_images = 100
num_test_images = 10
num_val_images = 10

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

key, subkey = random.split(key)
validation_X, validation_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_val_images, warmup_steps=warmup_steps)

key, subkey = random.split(key)
test_X, test_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_test_images, warmup_steps=warmup_steps)

key, subkey = random.split(key)
train_X, train_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_train_images, outfile=data_gif, warmup_steps=warmup_steps)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(Ms=[3], ks=[0,1,2], parities=[0,1], D=D, operators=group_actions)

one_point = train_X.get_subset(jnp.array([0]))

if load_file:
    params = jnp.load(load_file)
else:
    key, subkey = random.split(key)
    params = ml.init_params(
        partial(batch_net, conv_filters=conv_filters), 
        one_point, 
        subkey,
    )
    print('Num params:', ml.count_params(params))

    optimizer = optax.adam(
        optax.exponential_decay(lr, transition_steps=int(train_X.get_L() / batch_size), decay_rate=decay)
    )

    params, train_loss, val_loss = ml.train(
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