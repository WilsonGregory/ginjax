import time
import argparse
import sys
from functools import partial
import itertools as it

import jax.numpy as jnp
from jax import random, vmap
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def quadratic(layer):
    images = layer.to_images()

    out_images = []
    for i,j in it.combinations_with_replacement(range(len(images)), 2):
        out_images.append(images[i] * images[j])

    return geom.Layer.from_images(out_images)

def batch_net(params, layer, key, train, conv_filters, return_params=False):
    img_k = list(layer.keys())[0][0]

    layer, params = ml.batch_conv_layer(params, layer, conv_filters, depth=1, mold_params=return_params)
    layer = geom.BatchLayer(vmap(quadratic)(layer).data, layer.D, layer.is_torus)

    layer, params = ml.batch_conv_layer(params, layer, conv_filters, depth=1, target_key=(img_k,0), mold_params=return_params)
    layer, params = ml.batch_cascading_contractions(params, layer, img_k, mold_params=return_params)
    layer, params = ml.batch_channel_collapse(params, layer, mold_params=return_params)

    return (layer, params) if return_params else layer

def map_and_loss(params, x, y, key, train, conv_filters,):
    # Run the neural network, then calculate the MSE loss with the expected output y
    return jnp.mean(vmap(ml.rmse_loss)(batch_net(params, x, key, train, conv_filters)[(1,0)], y[(1,0)]))

def target_function(x, conv_filters):
    prod = x.convolve_with(conv_filters[1]) * x.convolve_with(conv_filters[2])
    return (prod).convolve_with(conv_filters[3]).contract(0,1)

def handleArgs(argv):
  parser = argparse.ArgumentParser()
  parser.add_argument('-epochs', help='number of epochs to run', type=int, default=10)
  parser.add_argument('-lr', help='learning rate', type=float, default=0.1)
  parser.add_argument('-num_images', help='number of images', type=int, default=3)
  parser.add_argument('-v', '--verbose', help='how many status messages to print during training', type=int, default=1)

  args = parser.parse_args()

  return args.epochs, args.lr, args.num_images, args.verbose

# Main
epochs, learning_rate, num_images, verbose = handleArgs(sys.argv)

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
    return_type='list',
)
filter_layer = geom.Layer.from_images(conv_filters)

# construct num_images vector images, (N,N)
X = []
Y = []
for i in range(num_images):
    key, subkey = random.split(key)
    img = geom.GeometricImage(random.normal(subkey, shape=((N,)*D + (D,)*k)), 0, D)
    X.append(img)
    Y.append(target_function(img, conv_filters))

layer_x = geom.BatchLayer.from_images(X)
layer_y = geom.BatchLayer.from_images(Y)

one_point = geom.BatchLayer.from_images([X[0]])

key, subkey = random.split(key)
params = ml.init_params(partial(batch_net, conv_filters=filter_layer), one_point, subkey)
print('Number of params:', ml.count_params(params))

key, subkey = random.split(key)
params, _, _ = ml.train(
    layer_x,
    layer_y,
    partial(map_and_loss, conv_filters=filter_layer),
    params,
    subkey,
    ml.EpochStop(epochs, verbose=verbose),
    batch_size=1,
    optimizer=optax.adam(learning_rate),
)

print('train loss:', map_and_loss(params, layer_x, layer_y, None, None, filter_layer))

key, subkey = random.split(key)
X_test = [geom.GeometricImage(data, 0, D) for data in random.normal(subkey, shape=((num_images,) + (N,)*D + (D,)*k))]
Y_test = [target_function(image, conv_filters) for image in X_test]

layer_x_test = geom.BatchLayer.from_images(X_test)
layer_y_test = geom.BatchLayer.from_images(Y_test)

print('test loss:', map_and_loss(params, layer_x_test, layer_y_test, None, None, filter_layer))

