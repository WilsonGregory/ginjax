import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math

from jax import jit, random, value_and_grad, vmap, checkpoint
import jax.nn
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

SCALE = 'scale'
BIAS = 'bias'

CONV_FREE = 'free'
CONV_FIXED = 'fixed'

## GeometricImageNet Layers

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
    target_k=None, 
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
        target_k (int): only do that convolutions that can be contracted to this, defaults to None
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
    for k, prods_group in input_layer.items():
        if mold_params:
            this_params[k] = {}

        for filter_k, filter_group in conv_filters.items():
            if ((target_k is not None) and ((k + target_k - filter_k) % 2 != 0)):
                continue

            if mold_params:
                this_params[k][filter_k] = jnp.ones((len(filter_group), len(prods_group)))

            res_k = k + filter_k

            group_sums = vmap_sums(prods_group, this_params[k][filter_k])
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
            out_layer.append(res_k, res)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

@functools.partial(jit, static_argnums=[3,4])
def get_filter_block_from_invariants(params, input_layer, invariant_filters, out_depth, mold_params):
    """
    For each k in the input_layer and each k of the invariant filters, return a block of filters of shape
    (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of the input_layer.
    """
    vmap_sum = vmap(vmap(geom.linear_combination, in_axes=(None, 0)), in_axes=(None, 0))

    if (mold_params):
        params = {}

    filter_layer = {}
    for k, image_block in input_layer.items():
        filter_layer[k] = {}
        in_depth = image_block.shape[0]

        if (mold_params):
            params[k] = {}

        for filter_k, filter_block in invariant_filters.items():
            if (mold_params):
                params[k][filter_k] = jnp.ones((out_depth, in_depth, len(filter_block)))

            filter_layer[k][filter_k] = vmap_sum(filter_block, params[k][filter_k])

    return filter_layer, params

@functools.partial(jit, static_argnums=[2,3,4,5])
def get_filter_block(params, input_layer, M, out_depth, filter_k_set=None, mold_params=False):
    """
    For each k in the input_layer and each k of the invariant filters, form a filter block from params 
    of shape (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of the input_layer.
    """
    if filter_k_set is None:
        filter_k_set = { 0 }

    if mold_params:
        params = {}

    filter_layer = {}
    for k, image_block in input_layer.items():
        filter_layer[k] = {}
        in_depth = image_block.shape[0]

        if mold_params:
            params[k] = {}

        for filter_k in filter_k_set:
            if mold_params:
                params[k][filter_k] = jnp.ones((out_depth,in_depth) + (M,)*input_layer.D + (input_layer.D,)*filter_k)

            filter_layer[k][filter_k] = params[k][filter_k]
    
    return filter_layer, params

@functools.partial(checkpoint, static_argnums=(2,3,4,5,6,7,8,9,10,11))
def conv_layer_build_filters(
    params, 
    input_layer,
    filter_info, 
    depth,
    target_k=None, 
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
            filter_info['filter_k_set'],
            mold_params,
        )
        this_params[CONV_FREE] = filter_block_params
    else:
        raise Exception(f'conv_layer_build_filters: filter_info["type"] must be one of: raw, {CONV_FIXED}, {CONV_FREE}')
    
    layer = conv_layer_alt(
        input_layer, 
        filter_block, 
        target_k,
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
        for k,image_block in layer.items():
            if (mold_params):
                this_params[BIAS][k] = jnp.ones(depth)

            biased_image = vmap(lambda image,p: image + p)(image_block, this_params[BIAS][k]) #add a single scalar
            out_layer.append(k, biased_image)
    else:
        out_layer = layer

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_conv_layer(
    params, 
    input_layer, 
    filter_info, 
    depth,
    target_k=None, 
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
        target_k, 
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
        conv_filters (dictionary by k of jnp.array): conv filters we are using, (OIHWk) or
            alternatively (output_depth, input_depth, (N,)*D, (D,)*filter_k)
        input_layer (Layer): layer of the input images, (CHWk) where C is the input
            depth, or alternatively (depth, (N,)*D, (D,)*img_k)
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

    out_layer = input_layer.empty()
    for k, images_block in input_layer.items():
        for filter_k, filter_group in conv_filters[k].items():
            if ((target_k is not None) and ((k + target_k - filter_k) % 2 != 0)):
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
            out_layer.append(k + filter_k, convolved_images_block)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    return out_layer

def not_conv_layer(
    params,
    input_layer, 
    invariant_filters,
    depth,
    target_k=None, 
    max_k=None,
    # bias=None,
    mold_params=False,
):
    """
    This is a convolution layer where for each input pixel, the filter can be different. This means that
    it is not an identical function at each pixel, so it is not translation equivariant. However, by 
    constructing the filters from B_d-invariant filters, we can make a B_d-equivariant layer.
    """
    params_idx, this_params = get_layer_params(params, mold_params, NOT_CONV)

    N,D = input_layer.N, input_layer.D
    filter_block, filter_block_params = get_filter_block_from_invariants(
        this_params,
        input_layer,
        invariant_filters,
        depth * (N**D), #make a different filter per pixel (stride/dilations?)
        mold_params,
    )
    this_params[CONV_FIXED] = filter_block_params

    # map over filters
    vmap_not_convolve = vmap(geom.depth_not_convolve, in_axes=(None, None, 0, None))

    out_layer = input_layer.empty()
    for k, images_block in input_layer.items():
        for filter_k, filter_group_raw in filter_block[k].items():
            if ((target_k is not None) and ((k + target_k - filter_k) % 2 != 0)):
                continue

            #reshape filter block so it works with not_convolve
            filter_group = jnp.transpose(
                filter_group_raw.reshape((N**D, depth) + filter_group_raw.shape[1:]),
                (1,2,0) + tuple(range(3,D+filter_k+3)) 
            ) # split up depth and filter per pixel, then put it in (depth, in, filters, ...)

            convolved_images_block = vmap_not_convolve(
                input_layer.D, 
                images_block, 
                filter_group, 
                input_layer.is_torus,
                # stride, 
                # padding,
                # lhs_dilation, 
                # rhs_dilation,
            )
            out_layer.append(k + filter_k, convolved_images_block)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)


    # if bias: #is this equivariant?
    #     out_layer = layer.empty()
    #     if (mold_params):
    #         this_params[BIAS] = {}
    #     for k,image_block in layer.items():
    #         if (mold_params):
    #             this_params[BIAS][k] = jnp.ones(depth)

    #         biased_image = vmap(lambda image,p: image + p)(image_block, this_params[BIAS][k]) #add a single scalar
    #         out_layer.append(k, biased_image)
    # else:
    #     out_layer = layer

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_not_conv_layer(
    params,
    input_layer, 
    invariant_filters,
    depth,
    target_k=None, 
    max_k=None,
    # bias=None,
    mold_params=False,
):
    vmap_not_conv = vmap(not_conv_layer, in_axes=(None, 0, None, None, None, None, None))
    return vmap_not_conv(params, input_layer, invariant_filters, depth, target_k, max_k, mold_params)

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

@functools.partial(jit, static_argnums=1)
def activation_layer(layer, activation_function):
    scalar_layer = contract_to_scalars(layer)
    layer[0] = activation_function(scalar_layer[0])
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
    for k, img in layer.items():
        if (k > max_k):
            k_diff = k - max_k 
            k_diff += (k_diff % 2) #if its odd, we need to go one lower

            idx_shift = 1 + layer.D
            for contract_idx in geom.get_contraction_indices(k, k - k_diff):
                shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
                contract_img = geom.multicontract(img, shifted_idx)
                _, res_k = geom.parse_shape(contract_img.shape[1:], layer.D)

                out_layer.append(res_k, contract_img)
        else:
            out_layer.append(k, img)

    return out_layer

def contract_to_scalars(input_layer):
    suitable_images = input_layer.empty()
    for k, image_block in input_layer.items():
        if ((k % 2) == 0):
            suitable_images[k] = image_block 

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
        image_block = temp_layer[k]
        if mold_params:
            this_params[k] = {}

        idx_shift = 1 + input_layer.D # layer plus N x N x ... x N (D times)
        for u,v in it.combinations(range(idx_shift, k + idx_shift), 2):
            if mold_params:
                this_params[k][(u,v)] = jnp.ones(len(image_block))

            group_sum = jnp.expand_dims(geom.linear_combination(image_block, this_params[k][(u,v)]), axis=0)
            contracted_img = geom.multicontract(group_sum, ((u,v),))

            temp_layer.append(k-2, contracted_img)

    params = update_params(params, params_idx, this_params, mold_params)

    out_layer = temp_layer.empty()
    out_layer.append(target_k, temp_layer[target_k])
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
    for k, image_block in input_layer.items():
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
            shifted_idx = tuple((i + idx_shift, j + idx_shift) for i,j in contract_idx)
            contracted_img = geom.multicontract(image_block, shifted_idx)
            out_layer.append(target_k, contracted_img)

    return out_layer

def batch_all_contractions(target_k, input_layer):
    return vmap(all_contractions, in_axes=(None, 0))(target_k, input_layer)

@functools.partial(jit, static_argnums=[2,3,4])
def paramed_contractions(params, input_layer, target_k, depth, mold_params=False, contraction_maps=None):
    params_idx, this_params = get_layer_params(params, mold_params, PARAMED_CONTRACTIONS)
    D = input_layer.D

    out_layer = input_layer.empty()
    for k, image_block in input_layer.items():
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
            out_layer.append(target_k, image_block)
            continue

        N, _ = geom.parse_shape(image_block.shape[1:], D)
        if contraction_maps is None:
            maps = jnp.stack(
                [geom.get_contraction_map(input_layer.D, k, idxs) for idxs in geom.get_contraction_indices(k, target_k)]
            ) # (maps, out_size, in_size)
        else:
            maps = contraction_maps[k]

        if mold_params:
            this_params[k] = jnp.ones((depth, len(image_block), len(maps))) # (depth, channels, maps)

        def channel_contract(maps, p, image_block):
            # Given an image_block, contract in all the ways for each channel, then sum up the channels
            # maps.shape: (maps, out_tensor_size, in_tensor_size)
            # p.shape: (channels, maps)
            # image_block.shape: (channels, (N,)*D, (D,)*k)

            map_sum = vmap(geom.linear_combination, in_axes=(None, 0))(maps, p) #(channels, out_size, in_size)
            image_block.reshape((len(image_block), (N**D), (D**k)))
            vmap_contract = vmap(geom.apply_contraction_map, in_axes=(None, 0, 0, None))
            return jnp.sum(vmap_contract(D, image_block, map_sum, target_k), axis=0)

        vmap_contract = vmap(channel_contract, in_axes=(None, 0, None)) #vmap over depth in params
        depth_block = vmap_contract(
            maps, 
            this_params[k], 
            image_block, 
        ) #(depth, image_shape)

        out_layer.append(target_k, depth_block)

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
    for k in input_layer.keys():
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
    for k, image_block in input_layer.items():
        if (mold_params):
            this_params[k] = jnp.ones((depth, len(image_block)))

        out_layer.append(k, vmap(geom.linear_combination, in_axes=(None, 0))(image_block, this_params[k]))

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_channel_collapse(params, input_layer, depth=1, mold_params=False):
    vmap_channel_collapse = vmap(channel_collapse, in_axes=(None, 0, None, None), out_axes=(0, None))
    return vmap_channel_collapse(params, input_layer, depth, mold_params)

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
    for k, image_block in batch_layer.items():
        if mold_params:
            num_channels = image_block.shape[1]
            this_params[k] = { SCALE: jnp.ones(num_channels), BIAS: jnp.ones(num_channels) }

        if (train):
            mean = jnp.mean(image_block, axis=0) # shape (channels, (N,)*D, (D,)*k)
            var = jnp.var(image_block, axis=0) # shape (channels, (N,)*D, (D,)*k)

            if ((k in running_mean) and (k in running_var)):
                running_mean[k] = (1 - momentum)*running_mean[k] + momentum*mean
                running_var[k] = (1 - momentum)*running_var[k] + momentum*var
            else:
                running_mean[k] = mean
                running_var[k] = var                
        else: # not train, use the final value from training
            mean = running_mean[k]
            var = running_var[k]

        centered_scaled_image = (image_block - mean)/jnp.sqrt(var + eps)

        # Now we multiply each channel by a scalar, then add a bias to each channel.
        # This is following: https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm2d.html
        mult_images = vmap(vmap(lambda x,p: x * p), in_axes=(0,None))(centered_scaled_image, this_params[k][SCALE])
        added_images = vmap(vmap(lambda x,p: x + p), in_axes=(0,None))(mult_images, this_params[k][BIAS])
        out_layer[k] = added_images
    
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
    for k, image_block in input_layer.items():
        if mold_params:
            this_params[k] = { SCALE: jnp.ones(1) }

        num_image_idxs = 1 + input_layer.D # number of indices that are channel + spatial image indices
        # get the variance of the frobenius norm of the pixels
        norm = vmap_norm(input_layer.D, image_block) # (channel, (N,)*D)
        var = jnp.var(norm, axis=range(num_image_idxs), keepdims=True) # (1, (1,)*D)
        shaped_var = jnp.expand_dims(var, axis=range(len(var.shape), len(image_block.shape))) #(1, (1,)*D, (1,)*k)

        centered_scaled_image = image_block/jnp.sqrt(shaped_var + eps)

        centered_scaled_image *= this_params[k][SCALE]

        out_layer[k] = centered_scaled_image

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def batch_layer_norm(params, input_layer, eps=1e-05, mold_params=False):
    vmap_layer_norm = vmap(layer_norm, in_axes=(None, 0, None, None), out_axes=(0, None))
    return vmap_layer_norm(params, input_layer, eps, mold_params)

def max_pool_layer(input_layer, patch_len):
    return [image.max_pool(patch_len) for image in input_layer]

def average_pool_layer(input_layer, patch_len):
    return [image.average_pool(patch_len) for image in input_layer]

def unpool_layer(input_layer, patch_len):
    return [image.unpool(patch_len) for image in input_layer]

@jit
def integration_layer(input_layer, basis_layer):
    """
    input_layer and basis_layer must have the same number of channels
    args:
        basis (Layer): basis
    """
    assert list(input_layer.keys()) == [1] #assume this input is just a vector image
    assert list(basis_layer.keys()) == [1] #assume basis is a vector basis
    assert len(input_layer[1]) == len(basis_layer[1]) #input and basis must same number of channels

    out_vec_image = jnp.zeros(basis_layer[1].shape[1:])
    img = input_layer[1] 
    basis_block = basis_layer[1]

    vmap_mul = vmap(geom.mul, in_axes=(None, 0, 0))
    mul_out = vmap_mul(input_layer.D, img, basis_block)
    coeffs_image = geom.multicontract(mul_out, ((0,1),), idx_shift=input_layer.D+1)
    coeffs = jnp.sum(coeffs_image, axis=range(1, 1+input_layer.D)) #does this need to be translation equiv?

    out_vec_image = jnp.sum(vmap(lambda p,img: p*img)(coeffs, basis_block), axis=0, keepdims=True)

    out_layer = input_layer.empty()
    out_layer.append(1, out_vec_image)
    return out_layer, coeffs

def batch_integration_layer(input_layer, basis_layer):
    return vmap(integration_layer, in_axes=(0,None))(input_layer, basis_layer)

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
        CASCADING_CONTRACTIONS: cascading_contractions_init,
        PARAMED_CONTRACTIONS: paramed_contractions_init,
    }

    for k,v in override_initializers.items():
        initializers[k] = v 

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
    for k, inner_tree in tree.items():
        out_params[k] = { SCALE: jnp.ones(inner_tree[SCALE].shape), BIAS: jnp.zeros(inner_tree[BIAS].shape) }

    return out_params

def layer_norm_init(rand_key, tree):
    out_params = {}
    for k, inner_tree in tree.items():
        out_params[k] = { SCALE: jnp.ones(inner_tree[SCALE].shape) }

    return out_params

def channel_collapse_init(rand_key, tree):
    out_params = {}
    for k, params_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        bound = 1/jnp.sqrt(params_block.shape[1])
        out_params[k] = random.uniform(subkey, params_block.shape, minval=-bound, maxval=bound)

    return out_params

def conv_init(rand_key, tree, D):
    assert (CONV_FREE in tree) or (CONV_FIXED in tree)
    out_params = {}
    filter_type = CONV_FREE if CONV_FREE in tree else CONV_FIXED
    params = {}
    for k, d in tree[filter_type].items():
        params[k] = {}
        for filter_k, filter_block in d.items():
            rand_key, subkey = random.split(rand_key)
            bound = 1/jnp.sqrt(filter_block.shape[1]* (filter_block.shape[2]**D))
            params[k][filter_k] = random.uniform(subkey, shape=filter_block.shape, minval=-bound, maxval=bound)

    out_params[filter_type] = params

    if (BIAS in tree):
        bias_params = {}
        for k, params_block in tree[BIAS].items():
            # reuse the bound from above, it shouldn't be any different
            bias_params[k] = random.uniform(subkey, shape=params_block.shape, minval=-bound, maxval=bound)
        
        out_params[BIAS] = bias_params

    return out_params

def conv_old_init(rand_key, tree):
    # Keep this how it was originally initialized so old code still works the same.
    out_params = {}
    for k, d in tree.items():
        out_params[k] = {}
        for filter_k, params_block in d.items():
            rand_key, subkey = random.split(rand_key)
            out_params[k][filter_k] = 0.1*random.normal(subkey, shape=params_block.shape)

    return out_params

def cascading_contractions_init(rand_key, tree):
    out_params = {}
    for k, d in tree.items():
        out_params[k] = {}
        for contraction_idx, params_block in d.items():
            rand_key, subkey = random.split(rand_key)
            out_params[k][contraction_idx] = 0.1*random.normal(subkey, shape=params_block.shape)

    return out_params

def paramed_contractions_init(rand_key, tree):
    out_params = {}
    for k, param_block in tree.items():
        _, channels, maps = param_block.shape
        bound = 1/jnp.sqrt(channels * maps)
        rand_key, subkey = random.split(rand_key)
        out_params[k] = random.uniform(subkey, shape=param_block.shape, minval=-bound, maxval=bound)

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

def add_noise(X, stdev, rand_key):
    """
    Add mean 0, stdev standard deviation Gaussian noise to the data X.
    args:
        X (layer): the X input data to the model
        stdev (float): the standard deviation of the desired Gaussian noise
        rand_key (jnp.random key): the key for randomness
    """
    noisy_layer = {}
    for k, image_block in X.values():
        rand_key, subkey = random.split(rand_key)
        noisy_layer[k] = image_block + stdev*random.normal(subkey, shape=image_block.shape)

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
    """
    if (isinstance(stop_condition, ValLoss)):
        assert validation_X and validation_Y

    batch_loss_grad = value_and_grad(map_and_loss, has_aux=has_aux)

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

            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            epoch_loss += loss_val

        epoch_loss = epoch_loss / len(X_batches)
        train_loss.append(epoch_loss)

        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if (validation_X and validation_Y):
            rand_key, subkey = random.split(rand_key)
            X_val_batches, Y_val_batches = get_batch_layer(validation_X, validation_Y, batch_size, subkey)
            epoch_val_loss = 0
            for X_val_batch, Y_val_batch in zip(X_val_batches, Y_val_batches):
                rand_key, subkey = random.split(rand_key)
                if (has_aux):
                    one_val_loss, aux_data = map_and_loss(
                        params, 
                        X_val_batch, 
                        Y_val_batch, 
                        subkey,
                        train=False,
                        aux_data=aux_data,
                    )
                    epoch_val_loss += one_val_loss
                else:
                    epoch_val_loss += map_and_loss(params, X_val_batch, Y_val_batch, subkey, train=False)

            epoch_val_loss /= len(X_val_batches)
            val_loss.append(epoch_val_loss)

        if (save_params and ((epoch % 10) == 0)):
            jnp.save(save_params, stop_condition.best_params)

    if (has_aux):
        return stop_condition.best_params, aux_data, jnp.array(train_loss), jnp.array(val_loss)
    else:
        return stop_condition.best_params, jnp.array(train_loss), jnp.array(val_loss)