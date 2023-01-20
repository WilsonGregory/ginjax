import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random, value_and_grad
import jax.numpy as jnp
import optax

import geometricconvolutions.geometric as geom

## Layers

@functools.partial(jit, static_argnums=1)
def conv_layer(params, param_idx, conv_filters, input_layer, target_x=None):
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
                group_sum = geom.linear_combination(prods_group, params[param_idx:(param_idx + len(prods_group))])
                out_layer.append(group_sum.convolve_with(conv_filter))
                param_idx += len(prods_group)

    return out_layer, param_idx

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

def quad_fast(params, param_idx, images):
    image_dict = make_p_k_dict(images)

    out_layer = []
    for img in images:
        for parity in [0,1]:
            for k in image_dict[parity].keys():
                image_group = image_dict[parity][k]
                if (len(image_group) == 0):
                    continue

                group_sum = geom.linear_combination(image_group, params[param_idx:(param_idx + len(image_group))])
                out_layer.append(group_sum * img)
                param_idx += len(image_group)
        
    return out_layer, param_idx

@functools.partial(jit, static_argnums=[1,2,3])
def prod_layer(images, degree, max_k=None, with_replace=True):
    """
    For the given degree, apply that many prods in all possible combinations with replacement of the images.
    args:
        images (list of GeometricImage): images to prod, these should be all the convolved images
        degree (int): degree of the prods
        max_k (int): the max tensor order of the products, contractions are applied if necessary after each prod
        with_replace (bool): whether prods should be combos w/ replacement, or combos w/o replacement, defaults to true
    """
    if with_replace:
        idx_generator = it.combinations_with_replacement(range(len(images)), degree)
    else:
        idx_generator = it.combinations(range(len(images)), degree)

    prods = []
    for idxs in idx_generator: #multiply the images one at a time so we can limit k
        rolling_prods = [images[idxs[0]]]
        for idx in idxs[1:]:
            next_prods = []
            for prev_prod in rolling_prods:
                rolling_prod = prev_prod * images[idx]

                if (max_k and (rolling_prod.k > max_k)):
                    k_diff = rolling_prod.k - max_k 
                    if(k_diff % 2 == 1): #if its odd, we need to go one lower
                        k_diff += 1

                    for contract_idx in geom.get_contraction_indices(rolling_prod.k, rolling_prod.k - k_diff):
                        next_prods.append(rolling_prod.multicontract(contract_idx))
                else:
                    next_prods.append(rolling_prod)

            rolling_prods = next_prods

        prods.extend(rolling_prods)

    # print(len(prods))
    return prods

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

## Other

def rmse_loss(x, y, batch=True):
    """
    Root Mean Squared Error Loss, defaults to expecting BatchGeometricImages
    args:
        x (GeometricImage): the input image
        y (GeometricImage): the associated output for x that we are comparing against
        batch (bool): whether x and y are BatchGeometricImages, defaults to True
    """
    axes = tuple(range(1, len(x.shape()))) if batch else None
    rmse = jnp.sqrt(jnp.sum((x.data - y.data) ** 2, axis=axes))
    return jnp.mean(rmse) if batch else rmse

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

def get_batch_rollout(X, Y, batch_size, rand_key, rollout=1):
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
    data_len = len(X) - rollout
    assert batch_size <= data_len
    batch_indices = random.permutation(rand_key, data_len)

    X_batches = []
    Y_batches = []
    for i in range(int(math.ceil(data_len / batch_size))): #iterate through the batches of an epoch
        idxs = batch_indices[i*batch_size:(i+1)*batch_size]
        X_batches.append(geom.BatchGeometricImage.from_images(X, idxs))

        Y_batch_steps = []
        for step_size in range(1,rollout+1): #iterate through the number of steps we are rolling out
            Y_batch_steps.append(geom.BatchGeometricImage.from_images(Y[step_size:], idxs))

        Y_batches.append(Y_batch_steps)

    return X_batches, Y_batches

### Train

def train(
    X, 
    Y, 
    map_and_loss,
    params, 
    rand_key, 
    epochs, 
    batch_size=16, 
    learning_rate=0.1, 
    loss_steps=1, 
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
        learning rate (float or optax learning rate schedule): defaults to 0.1, the lr provided to optax Adam
        loss_steps (int): defaults to 1, the number of steps to rollout the prediction when computing the loss.
        save_params (str): defaults to None, where to save the params of the model, every epochs/10 th epoch.
        verbose (0,1,2 or 3): verbosity level. 3 prints loss every batch, 2 every epoch, 1 ever epochs/10 th epoch
            0 not at all.
    """
    assert verbose in {0,1,2,3}
    batch_loss_grad = value_and_grad(map_and_loss)

    optimizer = optax.adam(learning_rate)
    opt_state = optimizer.init(params)

    for i in range(epochs):
        rand_key, subkey = random.split(rand_key)

        X_batches, Y_batches = get_batch_rollout(X, Y, batch_size, subkey, loss_steps)
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

        if (verbose >= 2):
            print(f'Epoch {i}: {epoch_loss / len(X_batches)}')

    return params