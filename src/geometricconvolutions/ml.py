import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random, value_and_grad, vmap, checkpoint
import jax
import jax.numpy as jnp
import jax.debug
import optax

import geometricconvolutions.geometric as geom

## Constants

CONV_OLD = 'conv_old'
CONV = 'conv'
NOT_CONV = 'not_conv'
CHANNEL_COLLAPSE = 'collapse'
CASCADING_CONTRACTIONS = 'cascading_contractions'
PARAMED_CONTRACTIONS = 'paramed_contractions'
BATCH_NORM = 'batch_norm'
LAYER_NORM = 'layer_norm'
EQUIV_DENSE = 'equiv_dense'

SCALE = 'scale'
BIAS = 'bias'
SUM = 'sum'

CONV_FREE = 'free'
CONV_FIXED = 'fixed'

## GeometricImageNet Layers

# Old, consider removing
def add_to_layer(layer, k, image):
    if (k in layer):
        layer[k] = jnp.concatenate((layer[k], image))
    else:
        layer[k] = image

    return layer

@functools.partial(jit, static_argnums=[3,4,5,6,7,8,9])
def conv_layer(
    params, #non-static
    conv_filters, #non-static
    input_layer, #non-static
    target_key=None, 
    max_k=None,
    mold_params=False,
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
        conv_filters (dictionary by k of jnp.array): conv filters we are using
        input_layer (Layer): layer of the input images, can think of each image 
            as a channel in the traditional cnn case.
        target_key (2-tuple of ints): (k,parity) pair, only do that convolutions that can be contracted 
            to k and parity will match, defaults to None
        max_k (int): apply an order cap layer immediately following convolution, defaults to None

        # Below, these are all parameters that are passed to the convolve function.
        stride (tuple of ints): convolution stride
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D
    """
    params_idx, this_params = get_layer_params(params, mold_params, CONV_OLD)

    # map over dilations, then filters
    vmap_sums = vmap(geom.linear_combination, in_axes=(None, 0))
    vmap_convolve = vmap(geom.convolve, in_axes=(None, 0, 0, None, None, None, None, None))

    out_layer = input_layer.empty()
    for (k,parity), prods_group in input_layer.items():
        if mold_params:
            this_params[(k,parity)] = {}

        for (filter_k,filter_parity), filter_group in conv_filters.items():
            if (target_key is not None):
                if(
                    ((k + target_key[0] - filter_k) % 2 != 0) or #if resulting k cannot be contracted to desired
                    ((parity + filter_parity) % 2 != target_key[1]) #if resulting parity does not match desired
                ):
                    continue

            if mold_params:
                this_params[(k,parity)][(filter_k,filter_parity)] = jnp.ones((len(filter_group), len(prods_group)))

            res_k = k + filter_k

            group_sums = vmap_sums(prods_group, this_params[(k,parity)][(filter_k,filter_parity)])
            res = vmap_convolve(
                input_layer.D, 
                group_sums, 
                filter_group, 
                input_layer.is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )
            out_layer.append(res_k, (parity + filter_parity) % 2, res)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

@functools.partial(jit, static_argnums=[3,4])
def get_filter_block_from_invariants(params, input_layer, invariant_filters, out_depth, mold_params):
    """
    For each (k,parity) in the input_layer and each (k,parity) of the invariant filters, return a block
    of filters of shape (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of
    the input_layer.
    args:
        params (params tree): the learned params tree
        input_layer (Layer): input layer so that we know the input depth at each (k,parity) that we need
        invariant_filters (Layer): layer of invariant filters at each (k,parity) that are linearly combined
            to form the filter blocks that we apply.
        out_depth (int): the output depth of this conv layer
        mold_params (bool): True if we are building the params shape, defaults to False
    """
    vmap_sum = vmap(vmap(geom.linear_combination, in_axes=(None, 0)), in_axes=(None, 0))

    if (mold_params):
        params = {}

    filter_layer = {}
    for key, image_block in input_layer.items():
        filter_layer[key] = {}
        in_depth = image_block.shape[0]

        if (mold_params):
            params[key] = {}

        for filter_key, filter_block in invariant_filters.items():
            if (mold_params):
                params[key][filter_key] = jnp.ones((out_depth, in_depth, len(filter_block)))

            filter_layer[key][filter_key] = vmap_sum(filter_block, params[key][filter_key])

    return filter_layer, params

def get_filter_block_invariants2(params, input_layer, invariant_filters, target_keys, out_depth, mold_params):
    """
    Alternative get_filter_block function for conv_contract layers. In this case there is an input keys and
    specific targetted output keys, and we use the available invariant_filters to build the connections 
    between those two layers.
    args:
        params (params tree): the learned params tree
        input_layer (Layer): input layer so that we know the input depth at each (k,parity) that we need
        invariant_filters (Layer): available invariant filters of each (k,parity) that will be used 
        target_keys (tuple of (k,parity) tuples): targeted keys for the output layer
        out_depth (int): the output depth of this conv layer
        mold_params (bool): True if we are building the params shape, defaults to False
    """
    vmap_sum = vmap(vmap(geom.linear_combination, in_axes=(None, 0)), in_axes=(None, 0))

    if (mold_params):
        params = {}

    filter_layer = {}
    for key, image_block in input_layer.items():
        filter_layer[key] = {}
        in_depth = image_block.shape[0]

        if (mold_params):
            params[key] = {}

        for target_key in target_keys:
            k, parity = key
            target_k, target_parity = target_key
            if (k+target_k, (parity + target_parity) % 2) not in invariant_filters:
                continue # relevant when there isn't an N=3, (0,1) filter

            filter_block = invariant_filters[(k+target_k, (parity + target_parity) % 2)]
            if (mold_params):
                params[key][target_key] = jnp.ones((out_depth, in_depth, len(filter_block)))

            filter_layer[key][target_key] = vmap_sum(filter_block, params[key][target_key])

    return filter_layer, params

@functools.partial(jit, static_argnums=[2,3,4,5])
def get_filter_block(params, input_layer, M, out_depth, filter_key_set=None, mold_params=False):
    """
    For each (k,parity) in the input_layer and each (k,parity) of the invariant filters, form a filter 
    block from params of shape (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the 
    size of the input_layer.
    args:
        params (params tree): the learned params tree
        input_layer (Layer): input layer so that we know the input depth at each (k,parity) that we need
        M (int): the edge length of the filters
        out_depth (int): the output depth of this conv layer
        filter_key_set (frozenset of 2-tuples of (k,parity)): a set of the type of filters to build
        mold_params (bool): True if we are building the params shape, defaults to False
    """
    if filter_key_set is None:
        filter_key_set = { (0,0) }

    if mold_params:
        params = {}

    filter_layer = {}
    for key, image_block in input_layer.items():
        filter_layer[key] = {}
        in_depth = image_block.shape[0]

        if mold_params:
            params[key] = {}

        for (filter_k,filter_parity) in filter_key_set:
            if mold_params:
                data_block = jnp.ones((out_depth,in_depth) + (M,)*input_layer.D + (input_layer.D,)*filter_k)
                params[key][(filter_k,filter_parity)] = data_block

            filter_layer[key][(filter_k,filter_parity)] = params[key][(filter_k,filter_parity)]
    
    return filter_layer, params

def conv_layer_build_filters(
    params, 
    input_layer,
    filter_info, 
    depth,
    target_key=None, 
    max_k=None,
    bias=None,
    mold_params=False,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Wrapper for conv_layer_alt that constructs the filter_block from either invariant filters or 
    free parameters, i.e. regular convolution with fully learned filters. 
    """
    params_idx, this_params = get_layer_params(params, mold_params, CONV)

    if (isinstance(filter_info, geom.Layer)): #if just a layer is passed, defaults to fixed filters
        filter_block, filter_block_params = get_filter_block_from_invariants(
            this_params[CONV_FIXED], 
            input_layer, 
            filter_info, 
            depth,
            mold_params,
        )
        this_params[CONV_FIXED] = filter_block_params
    elif (filter_info['type'] == 'raw'):
        filter_block = filter_info['filters']
    elif (filter_info['type'] == CONV_FIXED):
        filter_block, filter_block_params = get_filter_block_from_invariants(
            this_params[CONV_FIXED], 
            input_layer, 
            filter_info['filters'], 
            depth,
            mold_params,
        )
        this_params[CONV_FIXED] = filter_block_params
    elif (filter_info['type'] == CONV_FREE):
        filter_block, filter_block_params = get_filter_block(
            this_params[CONV_FREE], 
            input_layer, 
            filter_info['M'],
            depth,
            frozenset(filter_info['filter_key_set']), # needs to be immutable to hash
            mold_params,
        )
        this_params[CONV_FREE] = filter_block_params
    else:
        raise Exception(f'conv_layer_build_filters: filter_info["type"] must be one of: raw, {CONV_FIXED}, {CONV_FREE}')
    
    layer = conv_layer_alt(
        input_layer, 
        filter_block, 
        target_key,
        max_k,
        stride,
        padding,
        lhs_dilation,
        rhs_dilation, 
    )
    if bias: #is this equivariant?
        out_layer = layer.empty()
        if (mold_params):
            this_params[BIAS] = {}
        for (k,parity), image_block in layer.items():
            if (mold_params):
                this_params[BIAS][(k,parity)] = jnp.ones(depth)

            biased_image = vmap(lambda image,p: image + p)(image_block, this_params[BIAS][(k,parity)]) #add a single scalar
            out_layer.append(k, parity, biased_image)
    else:
        out_layer = layer

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_conv_layer(
    params, 
    input_layer, 
    filter_info, 
    depth,
    target_key=None, 
    max_k=None,
    bias=None,
    mold_params=False,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Vmap wrapper for conv_layer_build_filters that maps over the batch in the input layer.
    """
    return vmap(conv_layer_build_filters, in_axes=((None,) + (0,) + (None,)*10), out_axes=(0, None))(
        params,
        input_layer, #vmapped over this arg
        filter_info, 
        depth,
        target_key, 
        max_k,
        bias,
        mold_params,
        stride, 
        padding,
        lhs_dilation, 
        rhs_dilation,
    )

@functools.partial(jit, static_argnums=[2,3,4,5,6,7])
def conv_layer_alt(
    input_layer, #non-static
    conv_filters, #non-static
    target_key=None, 
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
        conv_filters (dictionary by k of jnp.array): conv filters we are using, (OIHWk) or
            alternatively (output_depth, input_depth, (N,)*D, (D,)*filter_k)
        input_layer (Layer): layer of the input images, (CHWk) where C is the input
            depth, or alternatively (depth, (N,)*D, (D,)*img_k)
        target_key (2-tuple of ints): (k,parity) pair, only do that convolutions that can be contracted 
            to k and parity will match, defaults to None
        max_k (int): apply an order cap layer immediately following convolution, defaults to None

        # Below, these are all parameters that are passed to the convolve function.
        stride (tuple of ints): convolution stride
        padding (either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D
    """
    # map over filters
    vmap_convolve = vmap(geom.depth_convolve, in_axes=(None, None, 0, None, None, None, None, None))

    out_layer = input_layer.empty()
    for (k,parity), images_block in input_layer.items():
        for (filter_k,filter_parity), filter_group in conv_filters[(k,parity)].items():
            if (target_key is not None):
                if(
                    ((k + target_key[0] - filter_k) % 2 != 0) or #if resulting k cannot be contracted to desired
                    ((parity + filter_parity) % 2 != target_key[1]) #if resulting parity does not match desired
                ):
                    continue
            
            convolved_images_block = vmap_convolve(
                input_layer.D, 
                images_block, 
                filter_group, 
                input_layer.is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )
            out_layer.append(k + filter_k, (parity + filter_parity) % 2, convolved_images_block)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    return out_layer

@functools.partial(jit, static_argnums=[3,4,5,6,7,8,9,10])
def conv_contract(
    params, 
    input_layer,
    invariant_filters, 
    depth,
    target_keys, 
    bias=None,
    mold_params=False,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Per the theory, a linear map from kp -> k'p' can be characterized by a convolution with a
    (k+k')(pp') tensor filter, followed by k contractions. The filters are constructed from invariant
    filters.
    args:
        params (jnp.array): array of parameters, how learning will happen
        input_layer (Layer): layer of the input images, can think of each image 
            as a channel in the traditional cnn case.
        invariant_filters (Layer): conv filters we are using as a layer
        depth (int): number of output channels
        target_keys (tuple of (k,parity) tuples): the output (k,parity) types
        bias (bool): whether to include a bias image, defaults to None
        mold_params (bool): whether we are calculating the params tree or running the alg, defaults to False

        # Below, these are all parameters that are passed to the convolve function.
        stride (tuple of ints): convolution stride
        padding (either 'TORUS', 'VALID', 'SAME', or D length tuple of (upper,lower) pairs): 
        lhs_dilation (tuple of ints): amount of dilation to apply to image in each dimension D
        rhs_dilation (tuple of ints): amount of dilation to apply to filter in each dimension D
    """
    params_idx, this_params = get_layer_params(params, mold_params, CONV)
    filter_dict, filter_block_params = get_filter_block_invariants2(
        this_params[CONV_FIXED], 
        input_layer,
        invariant_filters,
        target_keys,
        depth, 
        mold_params,
    )
    this_params[CONV_FIXED] = filter_block_params

    # map over filters
    vmap_convolve = vmap(geom.depth_convolve_contract, in_axes=(None, None, 0, None, None, None, None, None))

    layer = input_layer.empty()
    for (k,parity), images_block in input_layer.items():
        for target_k, target_parity in target_keys:
            if (target_k, target_parity) not in filter_dict[(k,parity)]:
                continue
            
            filter_block = filter_dict[(k,parity)][(target_k, target_parity)]

            convolve_contracted_imgs = vmap_convolve(
                input_layer.D, 
                images_block, 
                filter_block, 
                input_layer.is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )

            layer.append(target_k, target_parity, convolve_contracted_imgs)

    if bias: #is this equivariant?
        out_layer = layer.empty()
        if (mold_params):
            this_params[BIAS] = {}
        for (k,parity), image_block in layer.items():
            if (mold_params):
                this_params[BIAS][(k,parity)] = jnp.ones(depth)

            biased_image = vmap(lambda image,p: image + p)(image_block, this_params[BIAS][(k,parity)]) #add a single scalar
            out_layer.append(k, parity, biased_image)
    else:
        out_layer = layer

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_conv_contract(
    params, 
    input_layer,
    invariant_filters, 
    depth,
    target_keys, 
    bias=None,
    mold_params=False,
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    return vmap(conv_contract, in_axes=((None, 0) + (None,)*9), out_axes=(0, None))(
        params, 
        input_layer, 
        invariant_filters, 
        depth, 
        target_keys, 
        bias, 
        mold_params, 
        stride, 
        padding,
        lhs_dilation,
        rhs_dilation,
    )

@functools.partial(jit, static_argnums=[3,4,5,6])
def not_conv_layer(
    params,
    input_layer, 
    invariant_filters,
    depth,
    target_key=None, 
    max_k=None,
    mold_params=False,
):
    """
    This is a convolution layer where for each input pixel, the filter can be different. This means that
    it is not an identical function at each pixel, so it is not translation equivariant. However, by 
    constructing the filters from B_d-invariant filters, we can make a B_d-equivariant layer.
    """
    params_idx, this_params = get_layer_params(params, mold_params, NOT_CONV)

    filter_block, filter_block_params = get_filter_block_from_invariants(
        this_params[CONV_FIXED],
        input_layer,
        invariant_filters,
        depth,
        mold_params,
    )
    this_params[CONV_FIXED] = filter_block_params

    # map over filters, out depth
    vmap_not_convolve = vmap(geom.depth_not_convolve, in_axes=(None, None, 0, None))

    out_layer = input_layer.empty()
    for (k,parity), images_block in input_layer.items():
        for (filter_k,filter_parity), filter_group in filter_block[(k,parity)].items():
            if (target_key is not None):
                if(
                    ((k + target_key[0] - filter_k) % 2 != 0) or #if resulting k cannot be contracted to desired
                    ((parity + filter_parity) % 2 != target_key[1]) #if resulting parity does not match desired
                ):
                    continue

            convolved_images_block = vmap_not_convolve(
                input_layer.D, 
                images_block, 
                filter_group, 
                input_layer.is_torus,
            )
            out_layer.append(k + filter_k, (parity + filter_parity) % 2, convolved_images_block)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_not_conv_layer(
    params,
    input_layer, 
    invariant_filters,
    depth,
    target_key=None, 
    max_k=None,
    mold_params=False,
):
    vmap_not_conv = vmap(not_conv_layer, in_axes=(None, 0, None, None, None, None, None), out_axes=(0,None))
    return vmap_not_conv(params, input_layer, invariant_filters, depth, target_key, max_k, mold_params)

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
    if (x.__class__ == geom.GeometricImage):
        return x.__class__.fill(x.spatial_dims, x.parity, x.D, fill, x.is_torus), param_idx

@functools.partial(jit, static_argnums=1)
def activation_layer(layer, activation_function):
    scalar_layer = contract_to_scalars(layer)
    for (k,parity), image_block in scalar_layer.items(): # k will 0
        layer[(k,parity)] = activation_function(image_block)

    return layer

@jit
def relu_layer(layer):
    return activation_layer(layer, jax.nn.relu)

def batch_relu_layer(layer):
    return vmap(relu_layer)(layer)

@functools.partial(jit, static_argnums=1)
def leaky_relu_layer(layer, negative_slope=0.01):
    return activation_layer(layer, functools.partial(jax.nn.leaky_relu, negative_slope=negative_slope))

def batch_leaky_relu_layer(layer, negative_slope=0.01):
    return vmap(leaky_relu_layer, in_axes=(0, None))(layer, negative_slope)

@jit
def sigmoid_layer(layer):
    return activation_layer(layer, jax.nn.sigmoid)

def kink(x, outer_slope=1, inner_slope=0):
    """
    An attempt to make a ReLU that is an odd function (i.e., kink(-x) = -kink(x)). Between -1 and 1, 
    kink scales the function by inner_slope, and outside that scales it by outer_slope.
    args:
        x (jnp.array): the values to perform the pointwise nonlinearity on
        outer_slope (float): slope for outer regions, defaults to 0.5
        inner_slope (float): slope for inner regions, defaults to 2
    """
    return jnp.where((x<=-0.5) | (x>=0.5), outer_slope*x, inner_slope*x)

def scalar_activation(layer, activation_function):
    """
    Given a layer, apply the nonlinear activation function to each scalar image_block. If the layer has
    odd parity, then the activation should be an odd function.
    """
    out_layer = layer.empty()
    for (k,parity), image_block in layer.items():
        out_image_block = activation_function(image_block) if (k == 0) else image_block
        out_layer.append(k, parity, out_image_block)

    return out_layer

def batch_scalar_activation(layer, activation_function):
    return vmap(scalar_activation, in_axes=(0, None))(layer, activation_function)

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
    print('WARNING: polynomial_layer does not work with current layer architecture')
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

def order_cap_layer(layer, max_k):
    """
    For each image with tensor order k larger than max_k, do all possible contractions to reduce it to order k, or k-1
    if necessary because the difference is odd.
    args:
        layer (Layer): the input images in the layer
        max_k (int): the max tensor order
    """
    out_layer = layer.empty()
    for (k,parity), img in layer.items():
        if (k > max_k):
            k_diff = k - max_k 
            k_diff += (k_diff % 2) #if its odd, we need to go one lower

            idx_shift = 1 + layer.D
            for contract_idx in geom.get_contraction_indices(k, k - k_diff):
                shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
                contract_img = geom.multicontract(img, shifted_idx)
                _, res_k = geom.parse_shape(contract_img.shape[1:], layer.D)

                out_layer.append(res_k, parity, contract_img)
        else:
            out_layer.append(k, parity, img)

    return out_layer

def contract_to_scalars(input_layer):
    suitable_images = input_layer.empty()
    for (k,parity), image_block in input_layer.items():
        if ((k % 2) == 0):
            suitable_images[(k,parity)] = image_block

    return all_contractions(0, suitable_images)

def cascading_contractions(params, input_layer, target_k, mold_params=False):
    """
    Starting with the highest k, sum all the images into a single image, perform all possible contractions,
    then add it to the layer below.
    args:
        params (list of floats): model params
        target_k (int): what tensor order you want to end up at
        input_layer (list of GeometricImages): images to contract
        mold_params (bool): if True, use jnp.ones as the params and keep track of their shape
    """
    params_idx, this_params = get_layer_params(params, mold_params, CASCADING_CONTRACTIONS)

    max_k = np.max(list(input_layer.keys()))
    temp_layer = input_layer.copy()
    for k in reversed(range(target_k+2, max_k+2, 2)):
        for parity in [0,1]:
            if (k,parity) not in temp_layer:
                continue 

            image_block = temp_layer[(k,parity)]
            if mold_params:
                this_params[(k,parity)] = {}

            idx_shift = 1 + input_layer.D # layer plus N x N x ... x N (D times)
            for u,v in it.combinations(range(idx_shift, k + idx_shift), 2):
                if mold_params:
                    this_params[(k,parity)][(u,v)] = jnp.ones(len(image_block))

                group_sum = jnp.expand_dims(
                    geom.linear_combination(image_block, this_params[(k,parity)][(u,v)]), 
                    axis=0,
                )
                contracted_img = geom.multicontract(group_sum, ((u,v),))

                temp_layer.append(k-2, parity, contracted_img)

    params = update_params(params, params_idx, this_params, mold_params)

    out_layer = temp_layer.empty()
    for parity in [0,1]:
        if (k,parity) in temp_layer:
            out_layer.append(target_k, parity, temp_layer[(target_k,parity)])

    return out_layer, params

def batch_cascading_contractions(params, input_layer, target_k, mold_params=False):
    return vmap(cascading_contractions, in_axes=(None, 0, None, None), out_axes=(0, None))(
        params,
        input_layer,
        target_k,
        mold_params,
    )

def all_contractions(target_k, input_layer):
    out_layer = input_layer.empty()
    for (k,parity), image_block in input_layer.items():
        idx_shift = 1 + input_layer.D # layer plus N x N x ... x N (D times)
        if ((k - target_k) % 2 != 0):
            print(
                'ml::all_contractions WARNING: Attempted contractions when input_layer is odd k away. '\
                'Use target_k parameter of the final conv_layer to prevent wasted convolutions.',
            )
            continue
        if (k < target_k):
            print(
                'ml::all_contractions WARNING: Attempted contractions when input_layer is smaller than '\
                'target_k. This means there may be wasted operations in the network.',
            ) #not actually sure the best way to resolve this
            continue

        for contract_idx in geom.get_contraction_indices(k, target_k):
            contracted_img = geom.multicontract(image_block, contract_idx, idx_shift=idx_shift)
            out_layer.append(target_k, parity, contracted_img)

    return out_layer

def batch_all_contractions(target_k, input_layer):
    return vmap(all_contractions, in_axes=(None, 0))(target_k, input_layer)

@functools.partial(jit, static_argnums=[2,3,4])
def paramed_contractions(params, input_layer, target_k, depth, mold_params=False, contraction_maps=None):
    params_idx, this_params = get_layer_params(params, mold_params, PARAMED_CONTRACTIONS)
    D = input_layer.D

    out_layer = input_layer.empty()
    for (k,parity), image_block in input_layer.items():
        if ((k - target_k) % 2 != 0):
            print(
                'ml::all_contractions WARNING: Attempted contractions when input_layer is odd k away. '\
                'Use target_k parameter of the final conv_layer to prevent wasted convolutions.',
            )
            continue
        if (k < target_k):
            print(
                'ml::all_contractions WARNING: Attempted contractions when input_layer is smaller than '\
                'target_k. This means there may be wasted operations in the network.',
            ) #not actually sure the best way to resolve this
            continue
        if (k == target_k):
            out_layer.append(target_k, parity, image_block)
            continue

        spatial_dims, _ = geom.parse_shape(image_block.shape[1:], D)
        spatial_size = np.multiply.reduce(spatial_dims)
        if contraction_maps is None:
            maps = jnp.stack(
                [geom.get_contraction_map(input_layer.D, k, idxs) for idxs in geom.get_contraction_indices(k, target_k)]
            ) # (maps, out_size, in_size)
        else:
            maps = contraction_maps[k]

        if mold_params:
            this_params[(k,parity)] = jnp.ones((depth, len(image_block), len(maps))) # (depth, channels, maps)

        def channel_contract(maps, p, image_block):
            # Given an image_block, contract in all the ways for each channel, then sum up the channels
            # maps.shape: (maps, out_tensor_size, in_tensor_size)
            # p.shape: (channels, maps)
            # image_block.shape: (channels, (N,)*D, (D,)*k)

            map_sum = vmap(geom.linear_combination, in_axes=(None, 0))(maps, p) #(channels, out_size, in_size)
            image_block.reshape((len(image_block), spatial_size, (D**k)))
            vmap_contract = vmap(geom.apply_contraction_map, in_axes=(None, 0, 0, None))
            return jnp.sum(vmap_contract(D, image_block, map_sum, target_k), axis=0)

        vmap_contract = vmap(channel_contract, in_axes=(None, 0, None)) #vmap over depth in params
        depth_block = vmap_contract(
            maps, 
            this_params[(k,parity)], 
            image_block, 
        ) #(depth, image_shape)

        out_layer.append(target_k, parity, depth_block)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_paramed_contractions(params, input_layer, target_k, depth, mold_params=False):
    vmap_paramed_contractions = vmap(
        paramed_contractions, 
        in_axes=(None, 0, None, None, None, None),
        out_axes=(0, None),
    )

    # do this here because we already cache it, don't want to do slow loop-unrolling by jitting it
    contraction_maps = {}
    for k,_ in input_layer.keys():
        if k < 2:
            continue

        contraction_maps[k] = jnp.stack(
            [geom.get_contraction_map(input_layer.D, k, idxs) for idxs in geom.get_contraction_indices(k, target_k)]
        )

    return vmap_paramed_contractions(params, input_layer, target_k, depth, mold_params, contraction_maps)

@functools.partial(jit, static_argnums=[2,3])
def channel_collapse(params, input_layer, depth=1, mold_params=False):
    """
    Combine multiple channels into depth number of channels. Often the final step before exiting a GI-Net.
    In some ways this is akin to a fully connected layer, where each channel image is an input.
    args:
        params (params dict): the usual
        input_layer (Layer): input layer whose channels we will take a parameterized linear combination of
        depth (int): output channel depth, defaults to 1
        mold_params (bool): 
    """
    params_idx, this_params = get_layer_params(params, mold_params, CHANNEL_COLLAPSE)

    out_layer = input_layer.empty()
    for (k,parity), image_block in input_layer.items():
        if (mold_params):
            this_params[(k,parity)] = jnp.ones((depth, len(image_block)))

        out_layer.append(
            k, 
            parity, 
            vmap(geom.linear_combination, in_axes=(None, 0))(image_block, this_params[(k,parity)]),
        )

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_channel_collapse(params, input_layer, depth=1, mold_params=False):
    vmap_channel_collapse = vmap(channel_collapse, in_axes=(None, 0, None, None), out_axes=(0, None))
    return vmap_channel_collapse(params, input_layer, depth, mold_params)

@functools.partial(jax.jit, static_argnums=3)
def equiv_dense_layer(params, input, basis, mold_params=False):
    """
    A dense layer with a specific basis of linear maps, rather than any possible linear map. This
    allows you to pass an equivariant basis so the whole layer is equivariant.
    args:
        params (params dict): a tree of params 
        input (jnp.array): array of data, shape (in_d,)
        basis (jnp.array): basis of linear maps, shape (basis_len,out_d,in_d)
    """
    params_idx, this_params = get_layer_params(params, mold_params, EQUIV_DENSE)

    if (mold_params):
        this_params[SUM] = jnp.ones((len(basis),1,1))

    equiv_map = jnp.sum(this_params[SUM] * basis, axis=0) # (out_d, in_d)
    output = equiv_map @ input

    params = update_params(params, params_idx, this_params, mold_params)

    return output, params

def batch_equiv_dense_layer(params, input, basis, mold_params=False):
    vmap_equiv_dense_layer = vmap(equiv_dense_layer, in_axes=(None,0,None,None), out_axes=(0, None))
    return vmap_equiv_dense_layer(params, input, basis, mold_params)

@functools.partial(jit, static_argnums=[2,5,6,7])
def batch_norm(params, batch_layer, train, running_mean, running_var, momentum=0.1, eps=1e-05, mold_params=False):
    """
    Batch norm, this may or may not be equivariant.
    args:
        params (jnp.array): array of learned params
        batch_layer (BatchLayer): layer to apply to batch norm on
        train (bool): whether it is training, in which case update the mean and var
        running_mean (dict of jnp.array): array of mean at each k
        running_var (dict of jnp.array): array of var at each k
        momentum (float): how much of the current batch stats to include in the mean and var
        eps (float): prevent val from being scaled to infinity when the variance is 0
        mold_params (bool): True if we are learning the params shape, defaults to False
    """
    params_idx, this_params = get_layer_params(params, mold_params, BATCH_NORM)

    if ((running_mean is None) and (running_var is None)):
        running_mean = {}
        running_var = {}

    out_layer = batch_layer.empty()
    for key, image_block in batch_layer.items():
        if mold_params:
            num_channels = image_block.shape[1]
            this_params[key] = { SCALE: jnp.ones(num_channels), BIAS: jnp.ones(num_channels) }

        if (train):
            mean = jnp.mean(image_block, axis=0) # shape (channels, (N,)*D, (D,)*k)
            var = jnp.var(image_block, axis=0) # shape (channels, (N,)*D, (D,)*k)

            if ((key in running_mean) and (key in running_var)):
                running_mean[key] = (1 - momentum)*running_mean[key] + momentum*mean
                running_var[key] = (1 - momentum)*running_var[key] + momentum*var
            else:
                running_mean[key] = mean
                running_var[key] = var                
        else: # not train, use the final value from training
            mean = running_mean[key]
            var = running_var[key]

        centered_scaled_image = (image_block - mean)/jnp.sqrt(var + eps)

        # Now we multiply each channel by a scalar, then add a bias to each channel.
        # This is following: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        mult_images = vmap(vmap(lambda x,p: x * p), in_axes=(0,None))(centered_scaled_image, this_params[key][SCALE])
        added_images = vmap(vmap(lambda x,p: x + p), in_axes=(0,None))(mult_images, this_params[key][BIAS])
        out_layer[key] = added_images
    
    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params, running_mean, running_var

@functools.partial(jit, static_argnums=[2,3])
def layer_norm(params, input_layer, eps=1e-05, mold_params=False):
    """
    Implementation of layer norm, just scaling by variance of the norm. There seems to be issues with this
    layer however, it often results in values being NaN.
    """
    print('WARNING layer_norm: This layer is causing problems, for experimental use only.')
    params_idx, this_params = get_layer_params(params, mold_params, LAYER_NORM)
    vmap_norm = vmap(geom.norm, in_axes=(None, 0)) #expects channel, (N,)*D

    out_layer = input_layer.empty()
    for key, image_block in input_layer.items():
        if mold_params:
            this_params[key] = { SCALE: jnp.ones(1) }

        num_image_idxs = 1 + input_layer.D # number of indices that are channel + spatial image indices
        # get the variance of the frobenius norm of the pixels
        norm = vmap_norm(input_layer.D, image_block) # (channel, (N,)*D)
        var = jnp.var(norm, axis=range(num_image_idxs), keepdims=True) # (1, (1,)*D)
        shaped_var = jnp.expand_dims(var, axis=range(len(var.shape), len(image_block.shape))) #(1, (1,)*D, (1,)*k)

        centered_scaled_image = image_block/jnp.sqrt(shaped_var + eps)

        centered_scaled_image *= this_params[key][SCALE]

        out_layer[key] = centered_scaled_image

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_layer_norm(params, input_layer, eps=1e-05, mold_params=False):
    vmap_layer_norm = vmap(layer_norm, in_axes=(None, 0, None, None), out_axes=(0, None))
    return vmap_layer_norm(params, input_layer, eps, mold_params)

def max_pool_layer(input_layer, patch_len):
    return [image.max_pool(patch_len) for image in input_layer]

@functools.partial(jit, static_argnums=1)
def average_pool_layer(input_layer, patch_len):
    out_layer = input_layer.empty()
    vmap_avg_pool = vmap(geom.average_pool, in_axes=(None, 0, None))
    for (k,parity), image_block in input_layer.items():
        out_layer.append(k, parity, vmap_avg_pool(input_layer.D, image_block, patch_len))

    return out_layer

def batch_average_pool(input_layer, patch_len):
    return vmap(average_pool_layer, in_axes=(0, None))(input_layer, patch_len)

def unpool_layer(input_layer, patch_len):
    return [image.unpool(patch_len) for image in input_layer]

@jit
def basis_average_layer(input_layer, basis):
    """
    Experimental layer that finds an average over each basis element to get a coefficient for that basis
    element.
    """
    # input_layer must be a only have (1,0)
    # basis must be (basis_len, (N,)*D, (D,))
    D = input_layer.D
    num_coeffs = basis.shape[0]

    def basis_prod(image, Qi):
        return jnp.mean(
            geom.multicontract(geom.mul(D, image, Qi), ((0,1),), idx_shift=D), 
            axis=range(D),
        )
    
    # outer (first in result) maps over basis, inner (second in result) maps over channels of image
    vmap_basis_prod = vmap(vmap(basis_prod, in_axes=(0,None)), in_axes=(None,0))
    coeffs = jnp.sum(vmap_basis_prod(input_layer[(1,0)], basis), axis=1) # (num_coeffs * D,)
    return coeffs.reshape((D,int(num_coeffs/D))).transpose() # (num_coeffs/D,D)

def batch_basis_average_layer(input_layer, basis):
    return vmap(basis_average_layer, in_axes=(0,None))(input_layer, basis) # (L, num_coeffs, D)

## Params

def get_layer_params(params, mold_params, layer_name):
    """
    Given a network params tree, create a key, value (empty defaultdict) for the next layer if
    mold_params is true, or return the next key and value from the tree if mold_params is False
    args:
        params (dict tree of jnp.array): the entire params tree for a neural network function
        mold_params (bool): whether the layer is building the params tree or using it
        layer_name (string): type of layer, currently just a label
    """
    if (mold_params):
        params_key_idx = (len(list(params.keys())), layer_name)
        this_params = defaultdict(lambda: None)
    else:
        params_key_idx = next(iter(params.keys()))
        this_params = params[params_key_idx]

    return params_key_idx, this_params

def update_params(params, params_idx, layer_params, mold_params):
    """
    If mold_params is true, save the layer_params at the slot params_idx, building up the
    params tree. If mold_params is false, we are consuming layers so pop that set of params
    args:
        params (dict tree of jnp.array): the entire params tree for a neural network function
        params_idx (tuple (int,str)): the key of the params that we are updating
        layer_params (dict tree of params): the shaped param if mold_params is True
        mold_params (bool): whether the layer is building the params tree or using it
    """
    # In mold_params, we are adding params one layer at a time, so we add it. When not in mold_params,
    # we are popping one set of params from the front each layer.
    if (mold_params):
        params[params_idx] = layer_params
    else:
        del params[params_idx]

    return params

def print_params(params, leading_tabs=''):
    """
    Print the params tree in a structured fashion.
    """
    print('{')
    for k,v in params.items():
        if isinstance(v, dict):
            print(f'{leading_tabs}{k}: ', end='')
            print_params(v, leading_tabs=leading_tabs + '\t')
        else:
            print(f'{leading_tabs}{k}: {v.shape}')
    print(leading_tabs + '}')

def count_params(params):
    """
    Count the total number of params in the params tree
    args:
        params (dict tree of params): the params of a neural network function
    """
    num_params = 0
    for v in params.values():
        num_params += count_params(v) if isinstance(v, dict) else v.size

    return num_params

def init_params(net_func, input_layer, rand_key, return_func=False, override_initializers={}):
    """
    Use this function to construct and initialize the tree of params used by the neural network function. The
    first argument should be a function that takes (params, input_layer, rand_key, train, return_params) as 
    arguments. Any other arguments should be provided already, possibly using functools.partial. When return_params
    is true, the function should return params as the last element of a tuple or list.
    args:
        net_func (function): neural network function
        input_layer (geom.Layer): One piece of data to give the initial shape, doesn't have to match batch size
        rand_key (rand key): key used both as input and for the initialization of the params (gets split)
        return_func (bool): if False, return params, if True return a func that takes a rand_key and returns 
            the params. Defaults to False.
        override_initializers (dict): Pass custom initializers with this dictionary. The key is the layer name
            and the value is a function that takes (rand_key, tree) and returns the tree of initialized params.
    """
    rand_key, subkey = random.split(rand_key)
    params = net_func(defaultdict(lambda: None), input_layer, subkey, True, return_params=True)[-1]

    initializers = {
        BATCH_NORM: batch_norm_init,
        LAYER_NORM: layer_norm_init,
        CHANNEL_COLLAPSE: channel_collapse_init,
        CONV: functools.partial(conv_init, D=input_layer.D),
        CONV_OLD: conv_old_init,
        NOT_CONV: functools.partial(not_conv_init, D=input_layer.D),
        CASCADING_CONTRACTIONS: cascading_contractions_init,
        PARAMED_CONTRACTIONS: paramed_contractions_init,
        EQUIV_DENSE: equiv_dense_init,
    }

    initializers = { **initializers, **override_initializers }

    if return_func:
        return lambda in_key: recursive_init_params(params, in_key, initializers)
    else:
        rand_key, subkey = random.split(rand_key)
        return recursive_init_params(params, subkey, initializers)

def recursive_init_params(params, rand_key, initializers):
    """
    Given a tree of params, initialize all the params according to the initializers. No longer recursive.
    args:
        params (dict tree of jnp.array): properly shaped dict tree
        rand_key (rand key): used for initializing the parameters
    """
    out_tree = {}
    for (i, layer_name), v in params.items():
        rand_key, subkey = random.split(rand_key)
        out_tree[(i, layer_name)] = initializers[layer_name](subkey, v)

    return out_tree

def batch_norm_init(rand_key, tree):
    out_params = {}
    for key, inner_tree in tree.items():
        out_params[key] = { SCALE: jnp.ones(inner_tree[SCALE].shape), BIAS: jnp.zeros(inner_tree[BIAS].shape) }

    return out_params

def layer_norm_init(rand_key, tree):
    out_params = {}
    for key, inner_tree in tree.items():
        out_params[key] = { SCALE: jnp.ones(inner_tree[SCALE].shape) }

    return out_params

def channel_collapse_init(rand_key, tree):
    out_params = {}
    for key, params_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        bound = 1/jnp.sqrt(params_block.shape[1])
        out_params[key] = random.uniform(subkey, params_block.shape, minval=-bound, maxval=bound)

    return out_params

def conv_init(rand_key, tree, D):
    assert (CONV_FREE in tree) or (CONV_FIXED in tree)
    out_params = {}
    filter_type = CONV_FREE if CONV_FREE in tree else CONV_FIXED
    params = {}
    for key, d in tree[filter_type].items():
        params[key] = {}
        for filter_key, filter_block in d.items():
            rand_key, subkey = random.split(rand_key)
            if filter_type == CONV_FIXED:
                bound = 1/jnp.sqrt(filter_block.shape[1]*filter_block.shape[2])
            else: # filter_type == CONV_FREE:
                bound = 1/jnp.sqrt(filter_block.shape[1]*(filter_block.shape[2]**D))

            params[key][filter_key] = random.uniform(subkey, shape=filter_block.shape, minval=-bound, maxval=bound)

    out_params[filter_type] = params

    if (BIAS in tree):
        bias_params = {}
        for key, params_block in tree[BIAS].items():
            # reuse the bound from above, it shouldn't be any different
            bias_params[key] = random.uniform(subkey, shape=params_block.shape, minval=-bound, maxval=bound)
        
        out_params[BIAS] = bias_params

    return out_params

def conv_old_init(rand_key, tree):
    # Keep this how it was originally initialized so old code still works the same.
    out_params = {}
    for key, d in tree.items():
        out_params[key] = {}
        for filter_key, params_block in d.items():
            rand_key, subkey = random.split(rand_key)
            out_params[key][filter_key] = 0.1*random.normal(subkey, shape=params_block.shape)

    return out_params

def not_conv_init(rand_key, tree, D):
    out_params = {}
    params = {}
    for key, d in tree[CONV_FIXED].items():
        params[key] = {}
        for filter_key, filter_block in d.items():
            rand_key, subkey = random.split(rand_key)
            bound = 1/jnp.sqrt(filter_block.shape[1]*(filter_block.shape[2]))
            params[key][filter_key] = random.uniform(subkey, shape=filter_block.shape, minval=-bound, maxval=bound)

    out_params[CONV_FIXED] = params

    return out_params

def cascading_contractions_init(rand_key, tree):
    out_params = {}
    for key, d in tree.items():
        out_params[key] = {}
        for contraction_idx, params_block in d.items():
            rand_key, subkey = random.split(rand_key)
            out_params[key][contraction_idx] = 0.1*random.normal(subkey, shape=params_block.shape)

    return out_params

def paramed_contractions_init(rand_key, tree):
    out_params = {}
    for key, param_block in tree.items():
        _, channels, maps = param_block.shape
        bound = 1/jnp.sqrt(channels * maps)
        rand_key, subkey = random.split(rand_key)
        out_params[key] = random.uniform(subkey, shape=param_block.shape, minval=-bound, maxval=bound)

    return out_params

def equiv_dense_init(rand_key, tree):
    out_params = {}
    for key, param_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        bound = 1./jnp.sqrt(param_block.size)
        out_params[key] = random.uniform(subkey, shape=param_block.shape, minval=-bound, maxval=bound)

    return out_params

## Losses

def rmse_loss(x, y):
    """
    Root Mean Squared Error Loss.
    args:
        x (jnp.array): the input image
        y (jnp.array): the associated output for x that we are comparing against
    """
    return jnp.sqrt(mse_loss(x, y))

def mse_loss(x, y):
    return jnp.mean((x - y) ** 2)

def l2_loss(x, y):
    return jnp.sqrt(l2_squared_loss(x, y))

def l2_squared_loss(x, y):
    return jnp.sum((x - y) ** 2)

## Data and Batching operations

def get_batch_layer(X, Y, batch_size, rand_key):
    # X is a layer, Y is a layer
    """
    Given X, Y, construct a random batch of batch size. Each Y_batch is a list of length rollout to allow for
    calculating the loss with each step of the rollout.
    args:
        X (BatchLayer): the input data
        Y (BatchLayer): the target output, will be same as X, unless noise was added to X
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness
        rollout (int): number of steps of rollout to do for Y
    """
    len_X = len(next(iter(X.values()))) # all must have the same length, so just check first
    batch_indices = random.permutation(rand_key, len_X)

    X_batches = []
    Y_batches = []
    for i in range(int(math.ceil(len_X / batch_size))): #iterate through the batches of an epoch
        idxs = batch_indices[i*batch_size:(i+1)*batch_size]
        X_batches.append(X.get_subset(idxs))
        Y_batches.append(Y.get_subset(idxs))

    return X_batches, Y_batches

def map_in_batches(
    map_and_loss, 
    params, 
    layer_X, 
    layer_Y, 
    batch_size, 
    rand_key, 
    train, 
    has_aux=False, 
    aux_data=None,
):
    """
    Runs map_and_loss for the entire layer_X, layer_Y, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons.
    """
    rand_key, subkey = random.split(rand_key)
    X_batches, Y_batches = get_batch_layer(layer_X, layer_Y, batch_size, subkey)
    total_loss = 0
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        rand_key, subkey = random.split(rand_key)
        if (has_aux):
            one_loss, aux_data = map_and_loss(
                params, 
                X_batch, 
                Y_batch, 
                subkey,
                train,
                aux_data=aux_data,
            )
            total_loss += one_loss
        else:
            total_loss += map_and_loss(params, X_batch, Y_batch, subkey, train=False)

    return total_loss / len(X_batches)

def add_noise(layer, stdev, rand_key):
    """
    Add mean 0, stdev standard deviation Gaussian noise to the data X.
    args:
        X (layer): the X input data to the model
        stdev (float): the standard deviation of the desired Gaussian noise
        rand_key (jnp.random key): the key for randomness
    """
    noisy_layer = layer.empty()
    for (k, parity), image_block in layer.items():
        rand_key, subkey = random.split(rand_key)
        noisy_layer.append(k, parity, image_block + stdev*random.normal(subkey, shape=image_block.shape))

    return noisy_layer

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
    has_aux=False,
    aux_data=None,
    checkpoint_kwargs=None,
):
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the Adam optimizer to learn the
    parameters the minimize the map_and_loss function. The params are returned.
    args:
        X (BatchLayer): The X input data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        Y (BatchLayer): The Y target data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss (function): function that takes in batch layers of X, Y, and maps X to Y_hat
            using params, then calculates the loss with Y.
        params (jnp.array): 
        rand_key (jnp.random key): key for randomness
        stop_condition (StopCondition): when to stop the training process, currently only 1 condition
            at a time
        batch_size (int): defaults to 16, the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer, defaults to adam(learning_rate=0.1)
        validation_X (BatchLayer): input data for a validation data set as a layer by k 
            of (images, channels, (N,)*D, (D,)*k)
        validation_Y (BatchLayer): target data for a validation data set as a layer by k 
            of (images, channels, (N,)*D, (D,)*k)
        noise_stdev (float): standard deviation for any noise to add to training data, defaults to None
        save_params (str): if string, save params every 10 epochs, defaults to None
        has_aux (bool): Passed to value_and_grad, specifies whether there is auxilliary data returned from
            map_and_loss. If true, this auxilliary data will be passed back in to map_and_loss with the
            name "aux_data". The last aux_data will also be returned from this function.
        aux_data (any): initial aux data passed in to map_and_loss when has_aux is true.
        checkpoint_kwargs (dict): dictionary of kwargs to pass to jax.checkpoint. If None, checkpoint will
            not be called, defaults to None.
    """
    if (isinstance(stop_condition, ValLoss)):
        assert validation_X and validation_Y

    if checkpoint_kwargs is None:
        batch_loss_grad = value_and_grad(map_and_loss, has_aux=has_aux)
    else:
        batch_loss_grad = checkpoint(value_and_grad(map_and_loss, has_aux=has_aux), **checkpoint_kwargs)

    if (optimizer is None):
        optimizer = optax.adam(0.1)

    opt_state = optimizer.init(params)

    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    train_loss = []
    val_loss = []
    while (not stop_condition.stop(params, epoch, epoch_loss, epoch_val_loss)):
        if noise_stdev:
            rand_key, subkey = random.split(rand_key)
            train_X = add_noise(X, noise_stdev, subkey)
        else:
            train_X = X

        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batch_layer(train_X, Y, batch_size, subkey)
        epoch_loss = 0
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            rand_key, subkey = random.split(rand_key)
            if (has_aux):
                (loss_val, aux_data), grads = batch_loss_grad(
                    params, 
                    X_batch, 
                    Y_batch, 
                    subkey, 
                    True,
                    aux_data=aux_data,
                )
            else:
                loss_val, grads = batch_loss_grad(params, X_batch, Y_batch, subkey, True)

            updates, opt_state = optimizer.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss_val

        epoch_loss = epoch_loss / len(X_batches)
        train_loss.append(epoch_loss)

        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if (validation_X and validation_Y):
            rand_key, subkey = random.split(rand_key)
            epoch_val_loss = map_in_batches(
                map_and_loss, 
                params, 
                validation_X, 
                validation_Y, 
                batch_size, 
                subkey, 
                False, # train
                has_aux,
                aux_data,
            )
            val_loss.append(epoch_val_loss)

        if (save_params and ((epoch % 10) == 0)):
            jnp.save(save_params, stop_condition.best_params)

    if (has_aux):
        return stop_condition.best_params, aux_data, jnp.array(train_loss), jnp.array(val_loss)
    else:
        return stop_condition.best_params, jnp.array(train_loss), jnp.array(val_loss)

def benchmark(
    get_data,
    models,
    rand_key, 
    benchmark,
    benchmark_range,
    num_trials=1,
    num_results=1,
):
    """
    Method to benchmark multiple models as a particular benchmark over the specified range.
    args:
        get_data (function): function that takes as its first argument the benchmark_value, and a rand_key
            as its second argument. It returns the data which later gets passed to model.
        models (list of tuples): the elements of the tuple are (str) model_name, and (func) model.
            Model is a function that takes data and a rand_key and returns either a single float score
            or an iterable of length num_results of float scores.
        rand_key (jnp.random key): key for randomness
        benchmark (str): the type of benchmarking to do
        benchmark_range (iterable): iterable of the benchmark values to range over
        num_trials (int): number of trials to run, defaults to 1
        num_results (int): the number of results that will come out of the model function. If num_results is
            greater than 1, it should be indexed by range(num_results)
    returns:
        an np.array of shape (trials, benchmark_range, models, num_results) with the results all filled in
    """
    results = np.zeros((num_trials, len(benchmark_range), len(models), num_results))
    for i in range(num_trials):
        for j, benchmark_val in enumerate(benchmark_range):

            rand_key, subkey = random.split(rand_key)
            data = get_data(benchmark_val, subkey)

            for k, (model_name, model) in enumerate(models):
                print(f'trial {i} {benchmark}: {benchmark_val:.4f} {model_name}')

                rand_key, subkey = random.split(rand_key)
                res = model(data, subkey)

                if num_results > 1:
                    for q in range(num_results):
                        results[i,j,k,q] = res[q]
                else:
                    results[i,j,k,0] = res

    return results
