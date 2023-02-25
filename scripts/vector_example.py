import numpy as np
import time
import argparse
import sys
from functools import partial

import jax.numpy as jnp
from jax import random, vmap
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def net(params, x, D, is_torus, conv_filters, return_params=False):
    _, img_k = geom.parse_shape(x.shape, D)
    layer = { img_k: jnp.expand_dims(x, axis=0) }
    conv_filters = ml.make_layer(conv_filters)

    layer, param_idx = ml.conv_layer(params, 0, conv_filters, layer, D, is_torus)
    layer, param_idx = ml.polynomial_layer(params, int(param_idx), layer, D, 2)

    layer, param_idx = ml.conv_layer(params, 0, conv_filters, layer, D, is_torus, img_k)
    layer, param_idx = ml.cascading_contractions(params, param_idx, img_k, layer, D)
    net_output = geom.linear_combination(layer, params[param_idx:(param_idx + len(layer))])
    param_idx += len(layer)

    return (net_output, param_idx) if return_params else net_output

def map_and_loss(params, x, y, conv_filters, D, is_torus):
    # Run the neural network, then calculate the MSE loss with the expected output y
    return ml.rmse_loss(net(params, x, D, is_torus, conv_filters), y)

def target_function(x, conv_filters):
    prod = x.convolve_with(conv_filters[1]) * x.convolve_with(conv_filters[2])
    return (prod).convolve_with(conv_filters[3]).contract(0,1)

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-epochs', help='number of epochs to run', type=int, default=10)
  parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
  parser.add_argument('-num_images', help='number of images', type=int, default=3)

  args = parser.parse_args()

  return args.epochs, args.lr, args.num_images

# Main
epochs, learning_rate, num_images = handleArgs(sys.argv)

key = random.PRNGKey(time.time_ns())

D = 2
N = 64 #image size
k = 1
M = 3  #filter image size

group_actions = geom.make_all_operators(D)
conv_filters = geom.get_invariant_filters(
    Ms=[M],
    ks=[0,1,2],
    parities=[0,1],
    D=D,
    operators=group_actions,
    return_list=True,
)

# construct num_images vector images, (N,N)
X = []
Y = []
for i in range(num_images):
    key, subkey = random.split(key)
    img = geom.GeometricImage(random.normal(subkey, shape=((N,)*D + (D,)*k)), 0, D)
    X.append(img)
    Y.append(target_function(img, conv_filters))

key, subkey = random.split(key)

huge_params = jnp.ones(ml.param_count(X[0], conv_filters, 2))
_, num_params = net(huge_params, X[0].data, D, True, conv_filters, return_params=True)
print('Number of params:', num_params)
params = random.normal(subkey, shape=(num_params,))

key, subkey = random.split(key)

params = ml.train(
    X,
    Y,
    partial(map_and_loss, conv_filters=conv_filters, D=D, is_torus=True),
    params,
    subkey,
    epochs=epochs,
    optimizer=optax.adam(learning_rate),
)

vmap_map_loss = vmap(
    partial(map_and_loss, conv_filters=conv_filters, D=X[0].D, is_torus=X[0].is_torus),
    in_axes=(None, 0, 0),
)
print('train loss:', jnp.mean(vmap_map_loss(
    params, 
    geom.BatchGeometricImage.from_images(X).data,
    geom.BatchGeometricImage.from_images(Y).data, 
)))

key, subkey = random.split(key)
X_test = geom.BatchGeometricImage(random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k)), 0, D)
Y_test = target_function(X_test, conv_filters)

print('test loss:', jnp.mean(vmap_map_loss(params, X_test.data, Y_test.data)))

