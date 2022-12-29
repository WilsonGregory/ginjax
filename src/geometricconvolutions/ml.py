import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random
import jax.numpy as jnp
from jax.tree_util import register_pytree_node_class

import geometricconvolutions.geometric as geom

## Layers

@jit
def conv_layer(x, conv_filters):
    """
    For each conv_filter, apply it to the image x and return the list
    args:
        x (GeometricImage): image that we are applying the filters to
        conv_filters (list of GeometricFilter): the conv_filters we are applying to the image
    """
    return [x.convolve_with(conv_filter) for conv_filter in conv_filters]

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

def prod_layer(images, degree):
    """
    For the given degree, apply that many prods in all possible combinations with replacement of the images.
    args:
        images (list of GeometricImage): images to prod, these should be all the convolved images
        degree (int): degree of the prods
    """
    prods = []
    for idxs in it.combinations_with_replacement(range(len(images)), degree):
        prods.append(functools.reduce(lambda u,v: u * v, [images[idx] for idx in idxs]))

    return prods

@jit
def linear_layer(conv_layer_out):
    """
    This function is purely for code clarity, in many cases it should get optimized away, (c1*A)
    """
    return conv_layer_out

@jit
def quadratic_layer(conv_layer_out):
    """
    Given the image convolved with all the conv filters, calculate all the quadratic combinations, (c1*A) x (c2*A)
    args:
        conv_layer_out (list of GeometricImages): should be the output of the original image with all the conv filters
    """
    return prod_layer(conv_layer_out, 2)

@functools.partial(jit, static_argnums=1)
def final_layer(params, param_idx, x, conv_filters, input_layer):
    """
    Given the conv filters and the input images, perform the final convolution such that the resulting image will have
    the same parity as x and k such that we can perform a whole number of contractions to return it to x.k. We
    optimize the process to minimize the number of convolutions that need to be performed by summing similary shaped
    images.

    For example, when calculating the quadratic terms we are given [(c1*A) x (c2*A), (c1*A) x (c3*A), ...] as the input
    layer, then we apply the final convolution c4*((c1*A) x (c2*A))
    args:
        params (list of floats): model params
        x (GeometricImage): image that we are putting through the model
        conv_filters (list of GeometricFilters): all the filters we can apply in this final layer
        input_layer (list of GeometricImages): linear, quadratic, cubic, etc. function image outputs before final layer
    """
    prods_dict = make_p_k_dict(input_layer, rollup_set={0,1})
    filters_dict = make_p_k_dict(conv_filters, filters=True)

    last_layer = []
    for parity in [0,1]:
        for k in prods_dict[parity].keys():
            prods_group = prods_dict[parity][k]
            filter_group = filters_dict[(parity + x.parity) % 2][(k + x.k) % 2]

            if (len(filter_group) == 0 or len(prods_group) == 0):
                continue

            for conv_filter in filter_group:
                group_sum = geom.linear_combination(
                    prods_group,
                    params[param_idx:(param_idx + len(prods_group))],
                )
                last_layer.append(group_sum.convolve_with(conv_filter))

                assert (last_layer[-1].k % 2) == (x.k % 2)
                assert (last_layer[-1].parity % 2) == (x.parity % 2)

                param_idx += len(prods_group)

    return last_layer, param_idx

def cascading_contractions(params, param_idx, x, input_layer):
    """
    Starting with the highest k, sum all the images into a single image, perform all possible contractions,
    then add it to the layer below.
    args:
        params (list of floats): model params
        x (GeometricImage): image that is going through the model
        input_layer (list of GeometricImages): images to contract
    """
    images_by_k = {}
    for img in input_layer:
        if img.k in images_by_k:
            images_by_k[img.k].append(img)
        else:
            images_by_k[img.k] = [img]

    descending_k_dict = dict(reversed(sorted(images_by_k.items())))

    final_list = []
    for k in descending_k_dict.keys():
        images = descending_k_dict[k]
        for u,v in it.combinations(range(k), 2):
            group_sum = geom.linear_combination(
                images,
                params[param_idx:(param_idx + len(images))],
            )
            contracted_img = group_sum.contract(u,v)
            if contracted_img.k == x.k: #done contracting, add to the the final list
                final_list.append(contracted_img)
            else: #add the the next layer
                descending_k_dict[contracted_img.k].append(contracted_img)

            param_idx += len(images)

    return final_list, param_idx

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
    Given Xs, Ys, construct a random batch of batch size. Xs and Ys are lists of lists of Geometric images. The
    outer-most list has length equal to the number of channels, and the inner list has length of the data
    args:
        X (list of list of GeometricImages): the input data
        Y (list of list of GeometricImages): the target output
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness, should be a subkey from random.split
    """
    assert len(Xs) == len(Ys) # possibly will be loosened in the future
    assert batch_size <= len(Xs[0])
    batch_indices = random.permutation(rand_key, len(Xs[0]))[:batch_size]

    X_batch_list = []
    Y_batch_list = []
    for X, Y in zip(Xs, Ys):
        X_batch_list.append(geom.BatchGeometricImage.from_images([X[idx] for idx in batch_indices]))
        Y_batch_list.append(geom.BatchGeometricImage.from_images([Y[idx] for idx in batch_indices]))

    return X_batch_list, Y_batch_list


def get_batch(X, Y, batch_size, rand_key):
    """
    Given X, Y, construct a random batch of batch size
    args:
        X (list of GeometricImages): the input data
        Y (list of GeometricImages): the target output
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness
    """
    X_batch, Y_batch = get_batch_channel([X], [Y], batch_size, rand_key)
    return X_batch[0], Y_batch[0]

