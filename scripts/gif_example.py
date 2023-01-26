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
    first_layer, param_idx = ml.conv_layer(params, 0, conv_filters, [x])
    second_layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, first_layer)
    poly_layer, param_idx = ml.polynomial_layer(params, int(param_idx), second_layer, poly_degree=2, bias=False)
    poly_layer = ml.leaky_relu_layer(poly_layer)

    final_layer, param_idx = ml.conv_layer(params, int(param_idx), conv_filters, poly_layer, x)
    contract_layer, param_idx = ml.cascading_contractions(params, int(param_idx), x, final_layer)
    contract_layer = ml.leaky_relu_layer(contract_layer)

    net_output = geom.linear_combination(contract_layer, params[param_idx:(param_idx + len(contract_layer))])
    param_idx += len(contract_layer)
    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y_steps, conv_filters):
    # Run x through the net, then return its loss with y
    loss = 0

    if isinstance(y_steps, list): #if loss_steps is greater than 1
        for y in y_steps:
            net_out = net(params, x, conv_filters)
            x = x + net_out
            loss += ml.rmse_loss(x, y)
    else:
        loss = ml.rmse_loss(x + net(params, x, conv_filters), y_steps)

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
    parser.add_argument(
        '-noise',
        help='standard deviation of Gaussian noise to add to training inputs', 
        type=float, 
        default=None,
    )
    parser.add_argument('-v', '--verbose', help='levels of print statements during training', type=int, default=1)

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
        args.verbose,
    )

#Main
args = handleArgs(sys.argv)
infile, outfile, epochs, lr, batch, loss_steps, seed, save_file, load_file, noise, verbose = args

key = random.PRNGKey(time.time_ns()) if (seed is None) else random.PRNGKey(seed)

ts = read_gif(infile)

D = 2
images = [geom.GeometricImage(frame, 0, D) for frame in ts]
train_X, train_Y = ml.get_timeseries_XY(images, loss_steps=loss_steps, circular=True)
test_images = train_X

# start with basic 3x3 scalar, vector, and 2nd order tensor images
group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0],
    parities=[0],
    D=D,
    operators=group_actions,
    return_list=True,
)

if load_file:
    params = jnp.load(load_file)
else:
    huge_params = jnp.ones(10000)
    
    _, num_params = net(
        huge_params, 
        geom.BatchGeometricImage.from_images(train_X[:batch]), 
        conv_filters, 
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
        batch_size=batch,
        optimizer=optax.adam(optax.exponential_decay(lr, transition_steps=24, decay_rate=0.98), b1=0.8, b2=0.99),
        save_params=save_file,
        noise_stdev=noise,
        verbose=verbose,
    ) 

if save_file:
    jnp.save(save_file, params)

img = test_images[-1]

print("Rollout Loss:")
frames = []
for i in range(len(test_images)):
    net_out = net(params, img, conv_filters)
    img = img + net_out
    print(f'Step {i}: {ml.rmse_loss(img, test_images[i], batch=False)}')
    frames.append(img.data)

iio.imwrite(outfile, jnp.stack(frames))

