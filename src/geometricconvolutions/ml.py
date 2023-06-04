import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random, value_and_grad, vmap
import jax.nn
import jax.numpy as jnp
import optax

import geometricconvolutions.geometric as geom

## GeometricImageNet Layers

def make_layer(images):
    images_by_k = defaultdict(list)
    for image in images:
        images_by_k[image.k].append(image)

    if isinstance(images[0], geom.BatchGeometricImage) or isinstance(images[0], geom.GeometricFilter):
        return { k: jnp.stack([image.data for image in image_list]) for k, image_list in images_by_k.items() }
    elif isinstance(images[0], geom.GeometricImage):
        return { k: jnp.stack([jnp.expand_dims(image.data, axis=0) for image in image_list]) for k, image_list in images_by_k.items() }
    
def add_to_layer(layer, k, image):
    if (k in layer):
        layer[k] = jnp.concatenate((layer[k], image))
    else:
        layer[k] = image

    return layer

def merge_layers(layer_a, layer_b):
    for k, image in layer_b.items():
        layer_a = add_to_layer(layer_a, k, image)

    return layer_a

@functools.partial(jit, static_argnums=[1,4,5,6,7,8,9,10,11])
def conv_layer(
    params, #non-static
    param_idx, 
    conv_filters, #non-static
    input_layer, #non-static
    D, 
    is_torus, 
    target_k=None, 
    max_k=None,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    conv_layer takes a layer of conv filters and a layer of images and convolves them all together, taking
    parameterized sums of the images prior to convolution to control memory explosion.

    args:
        params (jnp.array): array of parameters, how learning will happen
        param_idx (int): current index of the params
        conv_filters (dictionary by k of jnp.array): layer of the conv filters we are using
        input_layer (dictionary by k of jnp.array): layer of the input images, can think of each image 
            as a channel in the traditional cnn case.
        D (int): dimension of the images
        is_torus (bool): whether the images data is on the torus or not
        target_k (int): only do that convolutions that can be contracted to this, defaults to None
        max_k (int): apply an order cap layer immediately following convolution, defaults to None

        # Below, these are all parameters that are passed to the convolve function.
        stride (tuple of ints): convolution stride
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D
    """
    # map over dilations, then filters
    vmap_sums = vmap(geom.linear_combination, in_axes=(None, 0))
    vmap_convolve = vmap(geom.convolve, in_axes=(None, 0, 0, None, None, None, None, None))

    out_layer = {}
    for k in input_layer.keys():
        prods_group = input_layer[k]
        for filter_k, filter_group in conv_filters.items():
            if ((target_k is not None) and ((k + target_k - filter_k) % 2 != 0)):
                continue

            res_k = k + filter_k

            param_shape = (len(filter_group), len(prods_group))
            num_params = np.multiply.reduce(param_shape)
            group_sums = vmap_sums(
                prods_group, 
                params[param_idx:(param_idx + num_params)].reshape(param_shape),
            )
            param_idx += num_params
            res = vmap_convolve(
                D, 
                group_sums, 
                filter_group, 
                is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )
            out_layer = add_to_layer(out_layer, res_k, res)

    if (max_k is not None):
        out_layer = order_cap_layer(D, out_layer, max_k)

    return out_layer, param_idx

@functools.partial(jit, static_argnums=[1,4])
def get_filter_block_from_invariants(params, param_idx, input_layer, invariant_filters, out_depth):
    """
    For each k in the input_layer and each k of the invariant filters, return a block of filters of shape
    (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of the input_layer.
    """
    vmap_sum = vmap(vmap(geom.linear_combination, in_axes=(None, 0)), in_axes=(None, 0))

    filter_layer = {}
    for k, image_block in input_layer.items():
        filter_layer[k] = {}
        in_depth = image_block.shape[0]

        for filter_k, filter_block in invariant_filters.items():
            param_shape = (out_depth, in_depth, len(filter_block))
            param_size = out_depth * in_depth * len(filter_block)
            filter_sum = vmap_sum(filter_block, params[param_idx:param_idx + param_size].reshape(param_shape))
            param_idx += param_size

            filter_layer[k][filter_k] = filter_sum

    return filter_layer, param_idx 

@functools.partial(jit, static_argnums=[1,4,5,6,7,8,9,10,11,12])
def conv_layer_fixed_filters(
    params, #non-static
    param_idx,
    conv_filters, #non-static
    input_layer, #non-static
    D, 
    is_torus, 
    depth,
    target_k=None, 
    max_k=None,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Wrapper for conv_layer_alt that constructs the filter_block from the invariant filters in 
    conv_filters with the specified output depth.
    """
    filter_block, param_idx = get_filter_block_from_invariants(
        params, 
        int(param_idx), 
        input_layer, 
        conv_filters, 
        out_depth=depth,
    )
    layer = conv_layer_alt(
        filter_block, 
        input_layer, 
        D, 
        is_torus, 
        target_k,
        max_k,
        stride,
        padding,
        lhs_dilation,
        rhs_dilation, 
    )
    return layer, param_idx

@functools.partial(jit, static_argnums=[2,3,4,5,6,7,8,9])
def conv_layer_alt(
    conv_filters, #non-static
    input_layer, #non-static
    D, 
    is_torus, 
    target_k=None, 
    max_k=None,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    A more traditional take on convolution layer. The input_layer is a dictionary by k of batches of images.
    The conv_filters is a dictionary by k of groups of batches of filters, and that the number of groups
    will be the length of the batch of the output for that k.
    args:
        conv_filters (dictionary by k of jnp.array): layer of the conv filters we are using, (OIHWk) or
            alternatively (output_depth, input_depth, (N,)*D, (D,)*filter_k)
        input_layer (dictionary by k of jnp.array): layer of the input images, (CHWk) where C is the input
            depth, or alternatively (depth, (N,)*D, (D,)*img_k)
        D (int): dimension of the images
        is_torus (bool): whether the images data is on the torus or not
        target_k (int): only do that convolutions that can be contracted to this, defaults to None
        max_k (int): apply an order cap layer immediately following convolution, defaults to None

        # Below, these are all parameters that are passed to the convolve function.
        stride (tuple of ints): convolution stride
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D
    """
    # map over filters
    vmap_convolve = vmap(geom.depth_convolve, in_axes=(None, None, 0, None, None, None, None, None))

    out_layer = {}
    for k, images_block in input_layer.items():
        for filter_k, filter_group in conv_filters[k].items():
            if ((target_k is not None) and ((k + target_k - filter_k) % 2 != 0)):
                continue

            convolved_images_block = vmap_convolve(
                D, 
                images_block, 
                filter_group, 
                is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )
            out_layer = add_to_layer(out_layer, k + filter_k, convolved_images_block)

    if (max_k is not None):
        out_layer = order_cap_layer(D, out_layer, max_k)

    return out_layer

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

@functools.partial(jit, static_argnums=[1,2])
def activation_layer(layer, D, activation_function):
    scalar_image = contract_to_scalars(layer, D)
    layer = add_to_layer(layer, 0, activation_function(scalar_image))
    return layer

@functools.partial(jit, static_argnums=1)
def relu_layer(layer, D):
    return activation_layer(layer, D, jax.nn.relu)

@functools.partial(jit, static_argnums=[1,2])
def leaky_relu_layer(layer, D, negative_slope=0.01):
    return activation_layer(layer, D, functools.partial(jax.nn.leaky_relu, negative_slope=negative_slope))

@functools.partial(jit, static_argnums=1)
def sigmoid_layer(layer, D):
    return activation_layer(layer, D, jax.nn.sigmoid)

@functools.partial(jit, static_argnums=[1,3,4])
def polynomial_layer(params, param_idx, layer, D, poly_degree):
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
    prods_by_degree = defaultdict(dict)
    out_layer = {}

    vmap_sums = vmap(geom.linear_combination, in_axes=(None, 0))
    vmap_mul = vmap(geom.mul, in_axes=(None, 0, 0))

    prods_by_degree[1] = layer
    for degree in range(2, poly_degree + 1):
        prev_images_dict = prods_by_degree[degree - 1]

        for prods_group in prev_images_dict.values(): #for each similarly shaped image group in the rolling prods
            for mult_block in layer.values(): #multiply by each mult_block in the images

                param_shape = (len(mult_block), len(prods_group))
                num_params = np.multiply.reduce(param_shape)
                group_sums = vmap_sums(
                    prods_group, 
                    params[param_idx:(param_idx + num_params)].reshape(param_shape),
                )
                param_idx += num_params
                prod = vmap_mul(D, group_sums, mult_block)
                _, prod_k = geom.parse_shape(prod.shape[1:], D)

                prods_by_degree[degree] = add_to_layer(prods_by_degree[degree], prod_k, prod)
                out_layer = add_to_layer(out_layer, prod_k, prod)

    return out_layer, param_idx

def order_cap_layer(D, images, max_k):
    """
    For each image with tensor order k larger than max_k, do all possible contractions to reduce it to order k, or k-1
    if necessary because the difference is odd.
    args:
        images (list of GeometricImages): the input images in the layer
        max_k (int): the max tensor order
    """
    out_layer = {}
    for k, img in images.items():
        if (k > max_k):
            k_diff = k - max_k 
            k_diff += (k_diff % 2) #if its odd, we need to go one lower

            idx_shift = 1 + D
            for contract_idx in geom.get_contraction_indices(k, k - k_diff):
                shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
                contract_img = geom.multicontract(img, shifted_idx)
                _, res_k = geom.parse_shape(contract_img.shape[1:], D)

                out_layer = add_to_layer(out_layer, res_k, contract_img)
        else:
            out_layer = add_to_layer(out_layer, k, img)

    return out_layer

def contract_to_scalars(input_layer, D):
    suitable_images = {}
    for k, image_block in input_layer.items():
        if ((k % 2) == 0):
            suitable_images[k] = image_block 

    return all_contractions(0, suitable_images, D)

def cascading_contractions(params, param_idx, target_k, input_layer, D):
    """
    Starting with the highest k, sum all the images into a single image, perform all possible contractions,
    then add it to the layer below.
    args:
        params (list of floats): model params
        param_idx (int): index of current location in params
        target_k (int): what tensor order you want to end up at
        input_layer (list of GeometricImages): images to contract
        D (int): dimension of the images
    """
    max_k = np.max(list(input_layer.keys()))
    for k in reversed(range(target_k+2, max_k+2, 2)):
        images = input_layer[k]

        idx_shift = 1 + D # layer plus N x N x ... x N (D times)
        for u,v in it.combinations(range(idx_shift, k + idx_shift), 2):
            group_sum = jnp.expand_dims(
                geom.linear_combination(images, params[param_idx:(param_idx + len(images))]),
                axis=0,
            )
            contracted_img = geom.multicontract(group_sum, ((u,v),))
            param_idx += len(images)

            input_layer = add_to_layer(input_layer, k-2, contracted_img)

    return input_layer[target_k], param_idx

def all_contractions(target_k, input_layer, D):
    final_image = None
    for k, image_block in input_layer.items():
        idx_shift = 1 + D # layer plus N x N x ... x N (D times)
        for contract_idx in geom.get_contraction_indices(k, target_k):
            shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
            contracted_img = geom.multicontract(image_block, shifted_idx)
            if final_image is not None:
                final_image = jnp.concatenate((final_image, contracted_img))
            else:
                final_image = contracted_img

    return final_image


def max_pool_layer(input_layer, patch_len):
    return [image.max_pool(patch_len) for image in input_layer]

def average_pool_layer(input_layer, patch_len):
    return [image.average_pool(patch_len) for image in input_layer]

def unpool_layer(input_layer, patch_len):
    return [image.unpool(patch_len) for image in input_layer]

## Baseline Layers

@functools.partial(jit, static_argnums=[1,2,3,5,6])
def get_filter_block(params, param_idx, D, M, input_layer, out_depth, filter_k_set=None):
    """
    For each k in the input_layer and each k of the invariant filters, form a filter block from params 
    of shape (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of the input_layer.
    """
    if filter_k_set is None:
        filter_k_set = { 0 }

    filter_layer = {}
    for k, image_block in input_layer.items():
        filter_layer[k] = {}
        in_depth = image_block.shape[0]

        for filter_k in filter_k_set:
            filter_shape = (out_depth,in_depth) + (M,)*D + (D,)*filter_k
            filter_size = np.multiply.reduce(filter_shape)

            filter_layer[k][filter_k] = params[param_idx:param_idx + filter_size].reshape(filter_shape)
            param_idx += filter_size
    
    return filter_layer, param_idx

@functools.partial(jit, static_argnums=[1,3,4,5,6,7,8,9,10,11,12,13])
def conv_layer_free_filters(
    params, #non-static
    param_idx,
    input_layer, #non-static
    D, 
    is_torus, 
    depth,
    M,
    filter_k_set=None,
    target_k=None, 
    max_k=None,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Wrapper for conv_layer_alt that first constructs the free filter block from params with the specified
    side length M, filter_k, and output depth.
    """
    filter_block, param_idx = get_filter_block(
        params, 
        int(param_idx), 
        D,
        M,
        input_layer, 
        out_depth=depth,
        filter_k_set=filter_k_set,
    )
    layer = conv_layer_alt(
        filter_block, 
        input_layer, 
        D, 
        is_torus, 
        target_k,
        max_k,
        stride,
        padding,
        lhs_dilation,
        rhs_dilation, 
    )
    return layer, param_idx

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
    Root Mean Squared Error Loss.
    args:
        x (GeometricImage): the input image
        y (GeometricImage): the associated output for x that we are comparing against
    """
    return jnp.sqrt(jnp.sum((x - y) ** 2))

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

class StopCondition:
    def __init__(self, verbose=0) -> None:
        assert verbose in {0, 1, 2}
        self.best_params = None
        self.verbose = verbose

    def stop(self, params, current_epoch, train_loss, val_loss):
        pass

    def log_status(self, epoch, train_loss, val_loss):
        if (train_loss is not None):
            if (val_loss is not None):
                print(f'Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f}')
            else:
                print(f'Epoch {epoch} Train: {train_loss:.7f}')

class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self, epochs, verbose=0) -> None:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(self, params, current_epoch, train_loss, val_loss) -> bool:
        self.best_params = params

        if (
            self.verbose == 2 or
            (self.verbose == 1 and (current_epoch % (self.epochs // np.min([10,self.epochs])) == 0))
        ):
            self.log_status(current_epoch, train_loss, val_loss)

        return current_epoch >= self.epochs
    
class TrainLoss(StopCondition):
    # Stop when the training error stops improving after patience number of epochs.

    def __init__(self, patience=0, min_delta=0, verbose=0) -> None:
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(self, params, current_epoch, train_loss, val_loss) -> bool:
        if (train_loss is None):
            return False
        
        if (train_loss < (self.best_train_loss - self.min_delta)):
            self.best_train_loss = train_loss
            self.best_params = params
            self.epochs_since_best = 0

            if (self.verbose >= 1):
                self.log_status(current_epoch, train_loss, val_loss)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience

class ValLoss(StopCondition):
     # Stop when the validation error stops improving after patience number of epochs.

    def __init__(self, patience=0, min_delta=0, verbose=0) -> None:
        super(ValLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(self, params, current_epoch, train_loss, val_loss) -> bool:
        if (val_loss is None):
            return False
        
        if (val_loss < (self.best_val_loss - self.min_delta)):
            self.best_val_loss = val_loss
            self.best_params = params
            self.epochs_since_best = 0

            if (self.verbose >= 1):
                self.log_status(current_epoch, train_loss, val_loss)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience

def train(
    X, 
    Y, 
    map_and_loss,
    params, 
    rand_key, 
    stop_condition,
    batch_size=16, 
    optimizer=None,
    validation_X=None,
    validation_Y=None,
    noise_stdev=None, 
    save_params=None,
):
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the Adam optimizer to learn the
    parameters the minimize the map_and_loss function. The params are returned.
    args:
        X (list of GeometricImages): The X input data to the model
        Y (list of GeometricImages): The Y target data for the model
        map_and_loss (function): function that takes in params, X, and Y, and maps X to Y_hat
            using params, then calculates the loss with Y.
        params (jnp.array): 
        rand_key (jnp.random key): key for randomness
        stop_condition (StopCondition): when to stop the training process, currently only 1 condition
            at a time
        batch_size (int): defaults to 16, the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer, defaults to adam(learning_rate=0.1)
        validation_X (list of GeometricImages): input data for a validation data set
        validation_Y (list of GeometricImages): target data for a validation data set
        noise_stdev (float): standard deviation for any noise to add to training data, defaults to None
        save_params (str): if string, save params every 10 epochs, defaults to None
    """
    if (isinstance(stop_condition, ValLoss)):
        assert validation_X and validation_Y

    batch_loss_grad = vmap(value_and_grad(map_and_loss), in_axes=(None, 0, 0))

    if (optimizer is None):
        optimizer = optax.adam(0.1)

    opt_state = optimizer.init(params)

    if (validation_X and validation_Y):
        batch_validation_X = geom.BatchGeometricImage.from_images(validation_X)
        batch_validation_Y = geom.BatchGeometricImage.from_images(validation_Y)
    else:
        batch_validation_X = None
        batch_validation_Y = None

    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    train_loss = []
    val_loss = []
    while (not stop_condition.stop(params, epoch, epoch_loss, epoch_val_loss)):
        rand_key, subkey = random.split(rand_key)

        if noise_stdev:
            train_X = add_noise(X, noise_stdev, subkey)
            rand_key, subkey = random.split(rand_key)
        else:
            train_X = X

        X_batches, Y_batches = get_batch_rollout(train_X, Y, batch_size, subkey)
        epoch_loss = 0
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            loss_val, grads = batch_loss_grad(params, X_batch.data, Y_batch.data)
            grads = jnp.mean(grads, axis=0)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += jnp.mean(loss_val)

        epoch_loss = epoch_loss / len(X_batches)
        train_loss.append(epoch_loss)

        epoch += 1

        if (batch_validation_X and batch_validation_Y):
            epoch_val_loss = jnp.mean(batch_loss_grad(params, batch_validation_X.data, batch_validation_Y.data)[0])
            val_loss.append(epoch_val_loss)

        if (save_params and ((epoch % 10) == 0)):
            jnp.save(save_params, stop_condition.best_params)

    return stop_condition.best_params, np.array(train_loss), np.array(val_loss)