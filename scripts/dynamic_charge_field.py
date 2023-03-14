#generate gravitational field
import sys
import numpy as np
import itertools as it
from functools import partial
import argparse
import time
import matplotlib.pyplot as plt
import imageio

from jax import vmap
import jax.numpy as jnp
import jax.random as random
import jax.nn
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils

def get_initial_charges(num_charges, N, D, rand_key):
    return N*random.uniform(rand_key, shape=(num_charges, D))

def get_velocity_vector(loc, charge_loc, charge):
    vec = loc - charge_loc
    scaling = jnp.linalg.norm(vec) ** 3
    return (charge / scaling) * vec

def get_velocity_field(N, D, charges):
    pixel_idxs = jnp.array(list(it.product(range(N), repeat=D)), dtype=int)
    velocity_field = jnp.zeros((N,)*D + (D,))

    vmap_get_vv = vmap(get_velocity_vector, in_axes=(0, None, None)) #all locs, one charge

    for charge in charges:
        velocity_field = velocity_field + vmap_get_vv(pixel_idxs, charge, 1).reshape((N,)*D + (D,))

    return geom.GeometricImage(velocity_field, 0, D, is_torus=False)

def update_charges(charges, delta_t):
    get_net_velocity = vmap(get_velocity_vector, in_axes=(None, 0, None)) #single loc, all charges

    new_charges = []
    for i in range(len(charges)):
        velocities = get_net_velocity(charges[i], jnp.concatenate((charges[:i], charges[i+1:])), 1)
        assert velocities.shape == (len(charges) - 1, 2)
        net_velocity = jnp.sum(velocities, axis=0)
        assert net_velocity.shape == charges[i].shape == (2,)
        new_charges.append(charges[i] + delta_t * net_velocity)
    return jnp.stack(new_charges)

def Qtransform(vector_field, s):
    vector_field_norm = vector_field.norm()
    return geom.GeometricImage(
        (4*(jax.nn.sigmoid(vector_field_norm.data / s)-0.5)) / vector_field_norm.data, 
        0, 
        vector_field.D,
        is_torus=vector_field.is_torus,
    ) * vector_field

def get_data(N, D, num_charges, num_steps, delta_t, s, rand_key, num_images=1, outfile=None, warmup_steps=0):
    assert (not outfile) or (num_images == 1)

    initial_fields = []
    final_fields = []
    for _ in range(num_images):
        rand_key, subkey = random.split(rand_key)
        # generate charges, generally in the center so that they don't repel off the grid
        charges = get_initial_charges(num_charges, N/2, D, subkey) + jnp.array([int(N/4)]*D)
        for i in range(warmup_steps):
            charges = update_charges(charges, delta_t)

        initial_fields.append(Qtransform(get_velocity_field(N, D, charges), s))
        if outfile:
            utils.plot_image(initial_fields[-1])
            plt.savefig(f'{outfile}_{0}.png')
            plt.close()

        for i in range(1,num_steps+1):
            charges = update_charges(charges, delta_t)
            if outfile:
                utils.plot_image(Qtransform(get_velocity_field(N, D, charges), s))
                plt.savefig(f'{outfile}_{i}.png')
                plt.close()

        if outfile:
            with imageio.get_writer(f'{outfile}.gif', mode='I') as writer:
                for i in range(num_steps+1):
                    image = imageio.imread(f'{outfile}_{i}.png')
                    writer.append_data(image)

        final_fields.append(Qtransform(get_velocity_field(N, D, charges), s))

    return initial_fields, final_fields

def net(params, x, D, is_torus, conv_filters, target_img, return_params=False):
    # Note that the target image is only used for its shape
    _, img_k = geom.parse_shape(x.shape, D)
    _, target_k = geom.parse_shape(target_img.shape, D)
    layer = { img_k: jnp.expand_dims(x, axis=0) }
    conv_filters = ml.make_layer(conv_filters)
    max_k = 5

    param_idx = 0
    for dilation in [1,2,4,2,1,1,2,1]:
        layer, param_idx = ml.conv_layer(
            params, 
            int(param_idx), 
            conv_filters, 
            layer, 
            D, 
            is_torus, 
            dilations=(dilation,), 
            max_k=max_k,
        )
        layer = ml.leaky_relu_layer(layer, D)

    layer, param_idx = ml.conv_layer(
        params, 
        int(param_idx),
        conv_filters,
        layer,
        D,
        is_torus,
        target_k, #final_layer, make sure out output is target_img shaped
    )
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
    parser.add_argument('--data_gif', help='where the save the gif of the charges moving', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('-d', '--decay', help='decay rate of learning rate', type=float, default=0.99)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.outfile,
        args.data_gif,
        args.epochs,
        args.lr,
        args.decay,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
outfile, data_gif, epochs, lr, decay, batch_size, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2
num_steps = 10 #number of steps to do
warmup_steps = 1 #number of initial steps to run. This ensures that particles don't start directly adjacent.
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
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[1,2],
    parities=[0],
    D=D,
    operators=group_actions,
    return_list=True,
)

one_point = train_X[0]

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(10000000)

    _, num_params = net(
        huge_params,
        one_point.data,
        one_point.D,
        one_point.is_torus,
        conv_filters,
        train_Y[0].data, #just used for shape
        return_params=True,
    )

    print('Num params:', num_params)

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    optimizer = optax.adam(
        optax.exponential_decay(lr, transition_steps=int(len(train_X) / batch_size), decay_rate=decay)
    )

    params = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, conv_filters=conv_filters, D=one_point.D, is_torus=one_point.is_torus),
        params,
        key,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optimizer,
        validation_X=validation_X,
        validation_Y=validation_Y,
        save_params=save_file,
        verbose=verbose,
    )

    if (save_file):
            jnp.save(save_file, params)

fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(32,12))

utils.plot_image(train_X[0], ax=axs[0,0])
utils.plot_image(train_Y[0], ax=axs[0,1])
train_img = geom.GeometricImage(
    net(params, one_point.data, one_point.D, one_point.is_torus, conv_filters, train_Y[0].data),
    train_X[0].parity,
    train_X[0].D,
    train_X[0].is_torus,
)
utils.plot_image(train_img, ax=axs[0,2])
utils.plot_image(train_Y[0] - train_img, ax=axs[0,3])

utils.plot_image(test_X[0], ax=axs[1,0])
utils.plot_image(test_Y[0], ax=axs[1,1])
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
test_img = geom.GeometricImage(
    net(params, test_X[0].data, test_X[0].D, test_X[0].is_torus, conv_filters, test_Y[0].data), 
    test_X[0].parity,
    test_X[0].D,
    test_X[0].is_torus,
)
utils.plot_image(test_img, ax=axs[1,2])
utils.plot_image(test_Y[0] - test_img, ax=axs[1,3])

plt.savefig(outfile)
