import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random, value_and_grad
import jax.nn
import jax.numpy as jnp
import optax

import geometricconvolutions.geometric as geom

## Layers

@functools.partial(jit, static_argnums=[1,5,6])
def conv_layer(params, param_idx, conv_filters, input_layer, target_x=None, bias=False, dilations=(None,)):
    """
    Perform all the conv_filters on each image of input_layer. For efficiency, we take parameterized linear
    combinations of like inputs (same k and parity) before applying the convolutions. This is equivalent to a fully
    connected layer.

    Alternatively, if target_x is provided, only do those convolutions which will result in the same parity as
    target_x and a tensor order k that can be contracted back to target_x.
    args:
        params (list of floats): model params
        param_idx (int): current index of where we are in the params array
        conv_filters (list of GeometricFilters): all the filters we can apply
        input_layer (list of GeometricImages): linear, quadratic, cubic, etc. function image outputs
        target_x (GeometricImage): defaults to None, image that we are trying to return to
        bias (bool): Whether to include a bias image, defaults to False
        dilations (list of ints): the dilation convolutions to perform, defaults to (None,) (equivalent to (1,))
    """
    prods_dict = make_p_k_dict(input_layer)
    filters_dict = make_p_k_dict(conv_filters, filters=True) if target_x else None

    out_layer = []
    for parity in [0,1]:
        for k in prods_dict[parity].keys():
            prods_group = prods_dict[parity][k]
            if (target_x):
                filter_group = filters_dict[(parity + target_x.parity) % 2][(k + target_x.k) % 2]
            else:
                filter_group = conv_filters

            if ((len(prods_group) == 0) or (len(filter_group) == 0)):
                continue

            for conv_filter in filter_group:
                for dilation in dilations:
                    if (bias and k == 0):
                        bias_img, param_idx = get_bias_image(params, param_idx, prods_group[0])
                        prods_group.append(bias_img)

                    group_sum = geom.linear_combination(prods_group, params[param_idx:(param_idx + len(prods_group))])
                    out_layer.append(group_sum.convolve_with(conv_filter, dilation=dilation))
                    param_idx += len(prods_group)

    return out_layer, param_idx

def get_bias_image(params, param_idx, x):
    """
    Get a constant image that acts as a bias in the network.
    args:
        params (jnp.array): the parameters
        param_idx (int): the current index of the parameters
        x (GeometricImage): the image the gives us the shape/parity/batchness of the bias image we are making
    """
    assert x.k == 0, "Currently only works for k=0 tensors in order to maintain equivariance"
    fill = params[param_idx:(param_idx + (x.D ** x.k))].reshape((x.D,)*x.k)
    param_idx += x.D ** x.k
    if (x.__class__ == geom.BatchGeometricImage):
        return x.__class__.fill(x.N, x.parity, x.D, fill, x.L, x.is_torus), param_idx
    elif (x.__class__ == geom.GeometricImage):
        return x.__class__.fill(x.N, x.parity, x.D, fill, x.is_torus), param_idx

def make_p_k_dict(images, filters=False, rollup_set={}):
    """
    Given a list of images, sort them into a dictionary of even/odd parity, then even/odd k. If they are not filters,
    instead do by the exact k. If k is in rollup_set, anticontract(2) to rollup to k+2.
    args:
        images (list of GeometricImages): images that we are sorting into the dict
        filters (bool): filters can be sorted into even/odd k, defaults to false
        rollup_set (set): values of k to rollup to k+2, only works for k=0,1 (see anticontract)
    """
    assert not (filters and len(rollup_set)) #filters shouldn't be rolled up b/c they get split into even/odd
    images_dict = { 0: defaultdict(list), 1: defaultdict(list) }

    for image in images:
        if filters:
            images_dict[image.parity % 2][image.k % 2].append(image)
        else:
            if image.k in rollup_set:
                image = image.anticontract(2)

            images_dict[image.parity % 2][image.k].append(image)

    return images_dict

@jit
def relu_layer(images):
    return [image.activation_function(jax.nn.relu) for image in images]

@jit
def leaky_relu_layer(images, negative_slope=0.01):
    leaky_relu = functools.partial(jax.nn.leaky_relu, negative_slope=negative_slope)
    return [image.activation_function(leaky_relu) for image in images]

@functools.partial(jit, static_argnums=[1,3,4])
def polynomial_layer(params, param_idx, images, poly_degree, bias=True):
    """
    Construct a polynomial layer for a given degree. Calculate the full polynomial up to that degree of all the images.
    For example, if poly_degree=3, calculate the linear, quadratic, and cubic terms.
    args:
        params (jnp.array): parameters
        param_idx (int): index of current location in params
        images (list of GeometricImages): the images to make the polynomial function
        poly_degree (int): the maximum degree of the polynomial 
        bias (bool): whether to include a constant bias image
    """
    out_layer_dict = defaultdict(list)

    out_layer_dict[0] = images
    for degree in range(1, poly_degree):
        prev_images_dict = make_p_k_dict(out_layer_dict[degree - 1])

        for img in images: #multiply by all the images
            for parity in [0,1]:
                for k in prev_images_dict[parity].keys():
                    image_group = prev_images_dict[parity][k]
                    if (len(image_group) == 0):
                        continue

                    if (bias and k == 0):
                        bias_img, param_idx = get_bias_image(params, param_idx, image_group[0])
                        image_group.append(bias_img)

                    group_sum = geom.linear_combination(image_group, params[param_idx:(param_idx + len(image_group))])
                    out_layer_dict[degree].append(group_sum * img)
                    param_idx += len(image_group)

    return list(it.chain(*list(out_layer_dict.values()))), param_idx

def order_cap_layer(images, max_k):
    """
    For each image with tensor order k larger than max_k, do all possible contractions to reduce it to order k, or k-1
    if necessary because the difference is odd.
    args:
        images (list of GeometricImages): the input images in the layer
        max_k (int): the max tensor order
    """
    out_layer = []
    for img in images:
        if (img.k > max_k):
            k_diff = img.k - max_k 
            if(k_diff % 2 == 1): #if its odd, we need to go one lower
                k_diff += 1

            for contract_idx in geom.get_contraction_indices(img.k, img.k - k_diff):
                out_layer.append(img.multicontract(contract_idx))
        else:
            out_layer.append(img)

    return out_layer

@functools.partial(jit, static_argnums=1)
def cascading_contractions(params, param_idx, x, input_layer):
    """
    Starting with the highest k, sum all the images into a single image, perform all possible contractions,
    then add it to the layer below.
    args:
        params (list of floats): model params
        param_idx (int): index of current location in params
        x (GeometricImage): image that is going through the model
        input_layer (list of GeometricImages): images to contract
    """
    images_by_k = defaultdict(list)
    max_k = np.max([img.k for img in input_layer])
    for img in input_layer:
        images_by_k[img.k].append(img)

    for k in reversed(range(x.k+2, max_k+2, 2)):
        images = images_by_k[k]

        for u,v in it.combinations(range(k), 2):
            group_sum = geom.linear_combination(images, params[param_idx:(param_idx + len(images))])
            contracted_img = group_sum.contract(u,v)
            images_by_k[contracted_img.k].append(contracted_img)

            param_idx += len(images)

    return images_by_k[x.k], param_idx

## Params

def param_count(x, conv_filters, deg):
    """
    Do some napkin math to figure out an upper bound on the number of paramaters that we will need.
    args:
        x (GeometricImage): the input to the net
        conv_filters (list of GeometricFilters): the set of convolutional filters we will be using
        deg (int): degree of the net, e.g. for quadratic it is 2
    """
    max_k = np.max([conv_filter.k for conv_filter in conv_filters])

    #combos w/ replacement, there are (n + deg -1) choose (deg)
    # final layer convolutions upper bound
    #vague idea of possible contractions, no way this is right

    return math.comb(len(conv_filters)+deg-1, deg) * len(conv_filters) * math.comb((max_k**(deg+1))+(x.k**deg), 2)

## Losses

def rmse_loss(x, y):
    """
    Root Mean Squared Error Loss, if x and y are both batches, the loss is per image.
    args:
        x (GeometricImage): the input image
        y (GeometricImage): the associated output for x that we are comparing against
    """
    assert isinstance(x, geom.BatchGeometricImage) == isinstance(y, geom.BatchGeometricImage)
    batch = isinstance(x, geom.BatchGeometricImage)
    axes = tuple(range(1, len(x.shape()))) if batch else None
    rmse = jnp.sqrt(jnp.sum((x.data - y.data) ** 2, axis=axes))
    return jnp.mean(rmse) if batch else rmse

## Data and Batching operations

def get_batch_channel(Xs, Ys, batch_size, rand_key):
    """
    Given Xs, Ys, construct random batches. All batches except the last batch will be of batch size, and the last
    batch could be smaller than batch_size if there aren't enough images to fill it. Xs and Ys are lists of lists of
    Geometric images. The outer-most list has length equal to the number of channels, and the inner list has length
    of the data. The output will be a list of channels, then a list of batches, then the batch itself which is a list
    of geometric images.
    args:
        X (list of list of GeometricImages): the input data
        Y_steps (list of list of list of GeometricImages): the target output
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness, should be a subkey from random.split
    """
    assert len(Xs) == len(Ys) # possibly will be loosened in the future
    data_len = len(Xs[0])
    assert batch_size <= data_len
    batch_indices = random.permutation(rand_key, data_len)

    X_batch_list = []
    Y_batch_list = []
    for X, Y_steps in zip(Xs, Ys): #iterate through the channels

        X_batches = []
        Y_batches = []
        for i in range(int(math.ceil(data_len / batch_size))): #iterate through the batches of an epoch
            idxs = batch_indices[i*batch_size:(i+1)*batch_size]
            X_batches.append(geom.BatchGeometricImage.from_images(X, idxs))

            Y_steps_batched = []
            for Y in Y_steps: #iterate through the number of steps we are rolling out
                Y_steps_batched.append(geom.BatchGeometricImage.from_images(Y, idxs))

            Y_batches.append(Y_steps_batched)

        X_batch_list.append(X_batches)
        Y_batch_list.append(Y_batches)

    return X_batch_list, Y_batch_list


def get_batch(X, Y_steps, batch_size, rand_key):
    """
    Given X, Y, construct a random batch of batch size
    args:
        X (list of GeometricImages): the input data
        Y (list of list of GeometricImages): the target output, in steps
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness
    """
    X_batch, Y_batch = get_batch_channel([X], [Y_steps], batch_size, rand_key)
    return X_batch[0], Y_batch[0]

def get_batch_rollout(X, Y, batch_size, rand_key):
    """
    Given X, Y, construct a random batch of batch size. Each Y_batch is a list of length rollout to allow for
    calculating the loss with each step of the rollout.
    args:
        X (list of GeometricImages): the input data
        Y (list of GeometricImages): the target output, will be same as X, unless noise was added to X
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness
        rollout (int): number of steps of rollout to do for Y
    """
    batch_indices = random.permutation(rand_key, len(X))

    X_batches = []
    Y_batches = []
    for i in range(int(math.ceil(len(X) / batch_size))): #iterate through the batches of an epoch
        idxs = batch_indices[i*batch_size:(i+1)*batch_size]
        X_batches.append(geom.BatchGeometricImage.from_images(X, idxs))

        if (isinstance(Y[0], list)): # there are multiple loss steps
            Y_batch_steps = []
            for step_size in range(len(Y[0])): #iterate through the number of steps we are rolling out
                Y_at_step = [y[step_size] for y in Y]
                Y_batch_steps.append(geom.BatchGeometricImage.from_images(Y_at_step, idxs))

            Y_batches.append(Y_batch_steps)
        else: #if there is just a single loss step            
            Y_batches.append(geom.BatchGeometricImage.from_images(Y, idxs))

    return X_batches, Y_batches

def add_noise(X, stdev, rand_key):
    """
    Add mean 0, stdev standard deviation Gaussian noise to the data X.
    args:
        X (list of GeometricImages): the X input data to the model
        stdev (float): the standard deviation of the desired Gaussian noise
        rand_key (jnp.random key): the key for randomness
    """
    noise = stdev*random.normal(rand_key, shape=(len(X),) + X[0].shape())
    return [x + geom.GeometricImage(noise, x.parity, x.D) for x, noise in zip(X, noise)]

def get_timeseries_XY(X, loss_steps=1, circular=False):
    """
    Given data X that is a time series, we want to form Y that is one step in the timeseries. If loss_steps is 1,
    then this function will return a list of GeometricImages. If loss steps is greater than 1, this function will
    return a list of lists where the inner list is the sequence of steps. If circular is true, then the end of the
    time series feeds directly into the beginning.
    args:
        X (list of GeometricImages): GeometricImages of the time series
        loss_steps (int): how many steps in our loss, defaults to 1
        circular (bool): whether the time series is circular, defaults to False
    """
    assert loss_steps >= 1
    data_len = len(X) if circular else (len(X) - loss_steps)

    Y = []
    for i in range(data_len):
        if (loss_steps == 1):
            Y.append(X[(i + 1) % len(X)])
        else:
            Y_steps = []
            for step_size in range(1, loss_steps + 1): #iterate through the number of steps we are rolling out
                Y_steps.append(X[(i + step_size) % len(X)])

            Y.append(Y_steps)

    return X[:data_len], Y

### Train

def train(
    X, 
    Y, 
    map_and_loss,
    params, 
    rand_key, 
    epochs, 
    batch_size=16, 
    optimizer=None,
    validation_X=None,
    validation_Y=None,
    noise_stdev=None, 
    save_params=None,
    verbose=1,
):
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the Adam optimizer to learn the
    parameters the minimize the map_and_loss function. The params are returned.
    args:
        X (list of GeometricImages): The X input data to the model
        Y (list of GeometricImages): The Y target data for the model
        map_and_loss (function): function that takes in params, X_batch, and Y_batch, maps X_batch to Y_batch_hat
            using params, then calculates the loss with Y_batch.
        params (jnp.array): 
        rand_key (jnp.random key): key for randomness
        epochs (int): number of epochs to run. An epoch is defined as a pass over the entire data, and may involve
            multiple batches.
        batch_size (int): defaults to 16, the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer, defaults to adam(learning_rate=0.1)
        validation_X (list of GeometricImages): input data for a validation data set
        validation_Y (list of GeometricImages): target data for a validation data set
        loss_steps (int): defaults to 1, the number of steps to rollout the prediction when computing the loss.
        noise_stdev (float): standard deviation for any noise to add to training data, defaults to None
        save_params (str): defaults to None, where to save the params of the model, every epochs/10 th epoch.
        verbose (0,1,2 or 3): verbosity level. 3 prints loss every batch, 2 every epoch, 1 ever epochs/10 th epoch
            0 not at all.
    """
    assert verbose in {0,1,2,3}
    batch_loss_grad = value_and_grad(map_and_loss)

    if (optimizer is None):
        optimizer = optax.adam(0.1)

    opt_state = optimizer.init(params)

    for i in range(epochs):
        rand_key, subkey = random.split(rand_key)

        if noise_stdev:
            train_X = add_noise(X, noise_stdev, subkey)
            rand_key, subkey = random.split(rand_key)
        else:
            train_X = X

        X_batches, Y_batches = get_batch_rollout(train_X, Y, batch_size, subkey)
        epoch_loss = 0
        for X_batch, Y_batch_steps in zip(X_batches, Y_batches):
            loss_val, grads = batch_loss_grad(params, X_batch, Y_batch_steps)
            if (verbose >= 3):
                print(f'Batch loss: {loss_val}')
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss_val

        if (i == 0 or ((i+1) % (epochs // np.min([10,epochs])) == 0)):
            if (save_params):
                jnp.save(save_params, params)
            if (verbose == 1):
                print(f'Epoch {i}: {epoch_loss / len(X_batches)}')
            if (validation_X and validation_Y):
                batch_validation_X = geom.BatchGeometricImage.from_images(validation_X)
                batch_validation_Y = geom.BatchGeometricImage.from_images(validation_Y)
                print('Validation Error: ', map_and_loss(params, batch_validation_X, batch_validation_Y))

        if (verbose >= 2):
            print(f'Epoch {i}: {epoch_loss / len(X_batches)}')

    return params