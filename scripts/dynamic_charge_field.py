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

    return geom.GeometricImage(velocity_field, 0, D)

def update_charges(charges, delta_t):
    get_net_velocity = vmap(get_velocity_vector, in_axes=(None, 0, None)) #single loc, all charges

    new_charges = []
    for i in range(len(charges)):
        velocities = get_net_velocity(charges[i], jnp.concatenate((charges[:i], charges[i+1:])), 1)
        net_velocity = jnp.sum(velocities)
        new_charges.append(charges[i] + delta_t * net_velocity)
        # jnp.sum(get_net_velocity(charges[i], charges[:i] + charges[i+1:], 1))
    return jnp.stack(new_charges)

def Qtransform(vector_field, s):
    vector_field_norm = vector_field.norm()
    return geom.GeometricImage(
        jnp.arcsinh(vector_field_norm.data / s) / vector_field_norm.data, 
        0, 
        vector_field.D,
    ) * vector_field

def get_data(N, D, num_charges, num_steps, delta_t, s, rand_key, num_images=1, outfile=None):
    assert (not outfile) or (num_images == 1)

    initial_fields = []
    final_fields = []
    for _ in range(num_images):
        rand_key, subkey = random.split(rand_key)
        # generate charges, generally in the center so that they don't repel off the grid
        charges = get_initial_charges(num_charges, N/2, D, subkey) + jnp.array([int(N/4)]*D)
        initial_fields.append(get_velocity_field(N, D, charges))

        for i in range(num_steps):
            charges = update_charges(charges, delta_t)
            if outfile:
                velocity_field = get_velocity_field(N, D, charges)
                utils.plot_image(velocity_field)
                plt.savefig(f'{outfile}_{i}.png')
                plt.close()

                utils.plot_image(Qtransform(velocity_field, s))
                plt.savefig(f'{outfile}_q_{i}.png')
                plt.close()


        if outfile:
            with imageio.get_writer(f'{outfile}.gif', mode='I') as writer:
                for i in range(num_steps):
                    image = imageio.imread(f'{outfile}_{i}.png')
                    writer.append_data(image)
            with imageio.get_writer(f'{outfile}_q.gif', mode='I') as writer:
                for i in range(num_steps):
                    image = imageio.imread(f'{outfile}_q_{i}.png')
                    writer.append_data(image)

        final_fields.append(get_velocity_field(N, D, charges))

    return initial_fields, final_fields

def net(params, x, conv_filters, target_img, return_params=False):
    # Note that the target image is only used for its shape
    layer, param_idx = ml.conv_layer(params, 0, conv_filters, [x])
    # layer, param_idx = ml.polynomial_layer(params, int(param_idx), layer, 2, bias=False)
    # layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=tuple(range(1,x.N)))
    # layer = ml.order_cap_layer(layer, max_k=4)
    # layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=(4,))
    # layer = ml.order_cap_layer(layer, max_k=4)
    # layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=(8,))
    # layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=(4,))
    # layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=(2,))
    # layer = ml.order_cap_layer(layer, max_k=4)

    layer, param_idx = ml.conv_layer(
        params, 
        int(param_idx),
        conv_filters,
        layer,
        target_img, #final_layer, make sure out output is target_img shaped
        # dilations=tuple(range(1, int(x.N/2))),
    )
    layer, param_idx = ml.cascading_contractions(params, int(param_idx), target_img, layer)

    net_output = geom.linear_combination(layer, params[param_idx:(param_idx + len(layer))])
    param_idx += len(layer)
    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y, conv_filters):
    # Run x through the net, then return its loss with y
    return ml.rmse_loss(net(params, x, conv_filters, y), y)

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('outfile', help='where to save the image', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

    args = parser.parse_args()

    return (
        args.outfile,
        args.epochs,
        args.lr,
        args.batch,
        args.seed,
        args.save,
        args.load,
        args.verbose,
    )

# Main
outfile, epochs, lr, batch_size, seed, save_file, load_file, verbose = handleArgs(sys.argv)

N = 16
D = 2
num_steps = 20
delta_t = 2
s = 0.4 # used for transforming the input/output data

num_points = 4
num_train_images = 1
num_test_images = 10
num_val_images = 5

key = random.PRNGKey(time.time_ns() if (seed is None) else seed)

key, subkey = random.split(key)
validation_X, validation_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_val_images)

key, subkey = random.split(key)
test_X, test_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_test_images)

key, subkey = random.split(key)
train_X, train_Y = get_data(N, D, num_points, num_steps, delta_t, s, subkey, num_train_images)

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[1,2],
    parities=[0,1],
    D=D,
    operators=group_actions,
    return_list=True,
)

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(100000)

    _, num_params = net(
        huge_params,
        geom.BatchGeometricImage.from_images(train_X),
        conv_filters,
        train_Y[0], #just used for shape
        return_params=True,
    )

    print('Num params:', num_params)

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    params = ml.train(
        train_X,
        train_Y,
        partial(map_and_loss, conv_filters=conv_filters),
        params,
        key,
        epochs=epochs,
        batch_size=batch_size,
        optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=int(len(train_X) / batch_size), decay_rate=0.995)),
        validation_X=validation_X,
        validation_Y=validation_Y,
        save_params=save_file,
        verbose=verbose,
    )

    if (save_file):
            jnp.save(save_file, params)

# fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24,12))

# utils.plot_image(train_X[0], ax=axs[0,0])
# utils.plot_image(train_Y[0], ax=axs[0,1])
# utils.plot_image(net(params, train_X[0], conv_filters, train_Y[0]), ax=axs[0,2])

# utils.plot_image(test_X[0], ax=axs[1,0])
# utils.plot_image(test_Y[0], ax=axs[1,1])
# test_loss = map_and_loss(
#     params, 
#     geom.BatchGeometricImage.from_images(test_X), 
#     geom.BatchGeometricImage.from_images(test_Y), 
#     conv_filters,
# )
# print('Full Test loss:', test_loss)
# print(f'One Test loss: {map_and_loss(params, test_X[0], test_Y[0], conv_filters)}')
# utils.plot_image(net(params, test_X[0], conv_filters, test_Y[0]), ax=axs[1,2])

# plt.savefig(outfile)