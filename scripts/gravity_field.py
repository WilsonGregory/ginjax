#generate gravitational field
import sys
import numpy as np
import itertools as it
from functools import partial
import argparse
import time
import matplotlib.pyplot as plt

import jax.numpy as jnp
import jax.random as random
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml
import geometricconvolutions.utils as utils

def get_gravity_vector(position1, position2, mass):
    r_vec = position1 - position2
    r_squared = np.linalg.norm(r_vec) ** 2
    return (mass / r_squared) * r_vec

def get_gravity_field_image(N, D, point_position, point_mass):
    field = np.zeros((N,)*D + (D,))

    # this could all be vectorized
    for position in it.product(range(N), repeat=D):
        position = np.array(position)
        if (np.all(position == point_position)):
            continue

        field[tuple(position)] = get_gravity_vector(point_position, position, point_mass)

    return geom.GeometricImage(field, 0, D, is_torus=False)

def get_data(N, D, num_points, rand_key, num_images=1):
    rand_key, subkey = random.split(rand_key)
    planets = random.uniform(subkey, shape=(num_points,))
    planets = planets / jnp.max(planets)

    masses = []
    gravity_fields = []
    for _ in range(num_images):
        point_mass = np.zeros((N,N))
        gravity_field = geom.GeometricImage.zeros(N=N, k=1, parity=0, D=D, is_torus=False)

        # Sample uniformly the cells
        rand_key, subkey = random.split(rand_key)
        possible_locations = np.array(list(it.product(range(N), repeat=D)))
        location_choices = random.choice(subkey, possible_locations, shape=(num_points,), replace=False, axis=0)
        for (x,y), mass in zip(location_choices, planets):
            point_mass[x,y] = mass
            gravity_field = gravity_field + get_gravity_field_image(N, D, np.array([x,y]), mass)

        masses.append(geom.GeometricImage(point_mass, 0, D, is_torus=False))
        gravity_fields.append(gravity_field)

    return masses, gravity_fields

def net(params, x, conv_filters, target_img, return_params=False):
    # Note that the target image is only used for its shape
    layer, param_idx = ml.conv_layer(params, 0, conv_filters, [x])
    layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, layer, dilations=tuple(range(1,x.N)))

    layer, param_idx = ml.conv_layer(
        params, 
        int(param_idx), 
        conv_filters, 
        layer, 
        target_img, #final_layer, make sure out output is target_img shaped
        dilations=tuple(range(1,int(x.N / 2))),
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

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(100000)

    _, num_params = net(
        huge_params,
        geom.BatchGeometricImage.from_images(train_X),
        conv_filters,
        train_Y[0],
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

fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(24,12))

utils.plot_image(train_X[0], ax=axs[0,0])
utils.plot_image(train_Y[0], ax=axs[0,1])
utils.plot_image(net(params, train_X[0], conv_filters, train_Y[0]), ax=axs[0,2])

utils.plot_image(test_X[0], ax=axs[1,0])
utils.plot_image(test_Y[0], ax=axs[1,1])
test_loss = map_and_loss(
    params, 
    geom.BatchGeometricImage.from_images(test_X), 
    geom.BatchGeometricImage.from_images(test_Y), 
    conv_filters,
)
print('Full Test loss:', test_loss)
print(f'One Test loss: {map_and_loss(params, test_X[0], test_Y[0], conv_filters)}')
utils.plot_image(net(params, test_X[0], conv_filters, test_Y[0]), ax=axs[1,2])

plt.savefig(outfile)
# plt.savefig('../images/gravity/gravity_test.png')