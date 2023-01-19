import sys
import time
import argparse
from functools import partial
import imageio.v3 as iio

import numpy as np
import jax.numpy as jnp
from jax import random
import optax
import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def read_gif(infile):
    return iio.imread(infile).astype('float32')[...,0] #only need 1 channel b/c its b/w

def net(params, x, conv_filters, return_params=False):
    first_layer_out, param_idx = ml.conv_layer(params, 0, conv_filters, [x])
    second_layer_out, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, first_layer_out)
    quad_layer_out, param_idx = ml.quad_fast(params, param_idx, second_layer_out)
    final_layer_out, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, quad_layer_out, x)
    contracted_images, param_idx = ml.cascading_contractions(params, int(param_idx), x, final_layer_out)

    net_output = geom.linear_combination(contracted_images, params[param_idx:(param_idx + len(contracted_images))])
    return (net_output, param_idx + len(contracted_images)) if return_params else net_output

def map_and_loss(params, x, y_steps, conv_filters):
    # Run x through the net, then return its loss with y
    loss = 0
    for y in y_steps:
        net_out = net(params, x, conv_filters)
        x = x + net_out
        loss += ml.rmse_loss(x, y)
    return loss

def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('infile', help='the gif file to import', type=str)
    parser.add_argument('outfile', help='the gif file to produce', type=str)
    parser.add_argument('-e', '--epochs', help='number of epochs to run', type=int, default=10)
    parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
    parser.add_argument('-batch', help='batch size', type=int, default=1)
    parser.add_argument('-loss_steps', help='how many rollout steps the loss should use', type=int, default=1)
    parser.add_argument('-seed', help='the random number seed', type=int, default=None)
    parser.add_argument('-s', '--save', help='file name to save the params', type=str, default=None)
    parser.add_argument('-l', '--load', help='file name to load params from', type=str, default=None)
    parser.add_argument('-noise', help='whether to add gaussian noise', default=False, action='store_true')

    args = parser.parse_args()

    return (
        args.infile,
        args.outfile,
        args.epochs,
        args.lr,
        args.batch,
        args.loss_steps,
        args.seed,
        args.save,
        args.load,
        args.noise,
    )

#Main
args = handleArgs(sys.argv)
infile, outfile, epochs, lr, batch, loss_steps, seed, save_file, load_file, noise = args

key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

ts = read_gif(infile)

D = 2
train_images = [geom.GeometricImage(frame, 0, D) for frame in ts]

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
    huge_params = jnp.ones(ml.param_count(train_images[0], conv_filters, 3))
    _, num_params = net(
        huge_params, 
        geom.BatchGeometricImage.from_images(train_images[:batch]), 
        conv_filters, 
        return_params=True,
    )

    print('Num params:', num_params)

    key, subkey = random.split(key)
    params = 0.1*random.normal(subkey, shape=(num_params,))

    if noise:
        # Add noise to help with longer rollouts
        noisy_train_images = []
        for x in train_images:
            key, subkey = random.split(key)
            noise = geom.GeometricImage(0.01*random.normal(subkey, shape=(x.shape())), x.parity, x.D)
            noisy_train_images.append(x + noise)

    params = ml.train(
        noisy_train_images if noise else train_images,
        train_images,
        partial(map_and_loss, conv_filters=conv_filters),
        params,
        key,
        epochs=epochs,
        batch_size=batch,
        learning_rate=optax.linear_schedule(lr, lr/10, 40),
        loss_steps=loss_steps,
        save_params=save_file,
    )

if save_file:
    jnp.save(save_file, params)

img = train_images[-1]

frames = []
for i in range(len(train_images)):
    net_out = net(params, img, conv_filters)
    img = img + net_out
    frames.append(img.data)

iio.imwrite(outfile, jnp.stack(frames))

