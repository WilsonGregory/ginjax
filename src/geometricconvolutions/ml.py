import itertools as it
import functools
from collections import defaultdict
import numpy as np
import math
import time

from jax import jit, random, value_and_grad, vmap, checkpoint
import jax
import jax.numpy as jnp
import jax.debug
import optax

import geometricconvolutions.geometric as geom

## Constants

CONV_OLD = 'conv_old'
CONV = 'conv'
CHANNEL_COLLAPSE = 'collapse'
CASCADING_CONTRACTIONS = 'cascading_contractions'
PARAMED_CONTRACTIONS = 'paramed_contractions'
BATCH_NORM = 'batch_norm'
GROUP_NORM = 'group_norm'
EQUIV_DENSE = 'equiv_dense'
VN_NONLINEAR = 'vector_neuron_nonlinear'
VECTOR_DOTS_NONLINEAR = 'vector_dots_nonlinear'
NORM_NONLINEAR = 'norm_nonlinear'

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
    parameterized sums of the images prior to convolution to control memory explosion. This is an old
    implementation, you should now use batch_conv_layer for a new method of parameterization, or 
    batch_conv_contract for the modern GI-Net technique of specifying the target out types.

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

    # this is old convolve, no batch, in_c, or out_c
    convolve_old = lambda img, ff: geom.convolve(input_layer.D, img[None,None], ff[None,None], input_layer.is_torus, stride, padding, lhs_dilation, rhs_dilation)[0,0]
    vmap_convolve = vmap(convolve_old)

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
            res = vmap_convolve(group_sums, filter_group)
            out_layer.append(res_k, (parity + filter_parity) % 2, res)

    if (max_k is not None):
        out_layer = order_cap_layer(out_layer, max_k)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

@functools.partial(jax.jit, static_argnums=[3,4,5])
def get_filter_block_from_invariants(params, input_layer, invariant_filters, target_keys, out_depth, mold_params):
    """
    For each (k,parity) of the input_layer and each (k,parity) of the target_keys, construct filters from
    the available invariant_filters to build all possible connections between those two layers.
    args:
        params (params tree): the learned params tree
        input_layer (BatchLayer): input layer so that we know the input depth at each (k,parity) that we need
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
        in_depth = image_block.shape[1] # this is an un-vmapped BatchLayer, so (batch,in_c,spatial,tensors)

        if (mold_params):
            params[key] = {}

        for target_key in target_keys:
            k, parity = key
            target_k, target_parity = target_key
            if (k+target_k, (parity + target_parity) % 2) not in invariant_filters:
                continue # relevant when there isn't an N=3, (0,1) filter

            filter_block = invariant_filters[(k+target_k, (parity + target_parity) % 2)]
            if (mold_params):
                params[key][target_key] = (
                    (out_depth, in_depth, len(filter_block)), # params_shape
                    (in_depth,) + filter_block.shape[1:1+input_layer.D+k], # shape to use for input bounds
                )
                params_block = jnp.ones((out_depth, in_depth, len(filter_block)))
            else:
                params_block = params[key][target_key]

            filter_layer[key][target_key] = vmap_sum(filter_block, params_block)

    return filter_layer, params

@functools.partial(jit, static_argnums=[2,3,4,5])
def get_filter_block(params, input_layer, M, target_keys, out_depth, mold_params=False):
    """
    For each (k,parity) of the input_layer and each (k,parity) of the target_keys, construct filters 
    to build all possible connections between those two layers. The filters are shape 
    (out_depth,in_depth, (M,)*D, (D,)*filter_k). Note that in_depth is the size of the input_layer.
    args:
        input_layer (BatchLayer): input layer so that we know the input depth at each (k,parity) that we need
        M (int): the edge length of the filters
        target_keys (tuple of tuple of ints): the target (k,parity) for this layer
        out_depth (int): the output depth of this conv layer
        mold_params (bool): True if we are building the params shape, defaults to False
    """
    D = input_layer.D
    if mold_params:
        params = {}

    filter_layer = {}
    for key, image_block in input_layer.items():
        filter_layer[key] = {}
        in_depth = image_block.shape[1]

        if mold_params:
            params[key] = {}

        for target_key in target_keys:
            filter_k = key[0] + target_key[0]
            if (mold_params):
                params[key][target_key] = (
                    (out_depth,in_depth) + (M,)*D + (D,)*filter_k, # params_shape
                    (in_depth,) + (M,)*D + (D,)*key[0], # shape to use for input bounds
                )
                params_block = jnp.ones(params[key][target_key][0])
            else:
                params_block = params[key][target_key]

            filter_layer[key][target_key] = params_block
    
    return filter_layer, params

def batch_conv_layer(
    params, 
    input_layer,
    filter_info, 
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
    Wrapper for conv_layer_alt that constructs the filter_block from either invariant filters or 
    free parameters, i.e. regular convolution with fully learned filters. 
    """
    params_idx, this_params = get_layer_params(params, mold_params, CONV)

    mean_bias = True
    if (isinstance(filter_info, geom.Layer)): #if just a layer is passed, defaults to fixed filters
        filter_block, filter_block_params = get_filter_block_from_invariants(
            this_params[CONV_FIXED], 
            input_layer, 
            filter_info, 
            target_keys,
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
            target_keys,
            depth,
            mold_params,
        )
        this_params[CONV_FIXED] = filter_block_params
    elif (filter_info['type'] == CONV_FREE):
        mean_bias = False
        filter_block, filter_block_params = get_filter_block(
            this_params[CONV_FREE],
            input_layer, 
            filter_info['M'], 
            target_keys, 
            depth,
            mold_params,
        )
        this_params[CONV_FREE] = filter_block_params
    else:
        raise Exception(f'conv_layer_build_filters: filter_info["type"] must be one of: raw, {CONV_FIXED}, {CONV_FREE}')
    
    layer = batch_conv_contract(
        input_layer,
        filter_block,
        target_keys,
        stride,
        padding,
        lhs_dilation,
        rhs_dilation,
    )

    if bias:
        layer, this_params = add_bias(this_params, layer, mean_bias, mold_params=mold_params)

    params = update_params(params, params_idx, this_params, mold_params)

    return layer, params

def add_bias(this_params, layer, mean_bias=True, mold_params=False):
    """
    Per-channel bias. To maintain equivariance, we add a scale of the mean to each layer.
    args:
        this_params (dict): this part of a params tree, not an entire thing. Since you are just adding
            this to the end of a layer.
        layer (BatchLayer): input layer
        mean_bias (bool): if true, add a scale of the mean, otherwise a single bias per channel, but it
            is the same across batches.
    """
    out_layer = layer.empty()
    if (mold_params):
        this_params[BIAS] = {}

    for (k,parity), image_block in layer.items():
        channels = image_block.shape[1]

        if (mold_params):
            this_params[BIAS][(k,parity)] = jnp.ones((1,channels) + (1,)*(image_block.ndim-2))

        if mean_bias:
            mean = jnp.mean(image_block, axis=tuple(range(2,2+layer.D)), keepdims=True)
            assert mean.shape == image_block.shape[:2] + (1,)*layer.D + (layer.D,)*k
            out_layer.append(k, parity, image_block + this_params[BIAS][(k,parity)]*mean)
        else:
            out_layer.append(k, parity, image_block + this_params[BIAS][(k,parity)])

    return out_layer, this_params

@functools.partial(jit, static_argnums=[2,3,4,5,6])
def batch_conv_contract(
    input_layer,
    conv_filters, 
    target_keys, 
    # Convolve kwargs that are passed directly along
    stride=None, 
    padding=None,
    lhs_dilation=None, 
    rhs_dilation=None,
):
    """
    Per the theory, a linear map from kp -> k'p' can be characterized by a convolution with a
    (k+k')(pp') tensor filter, followed by k contractions. 
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
    layer = input_layer.empty()
    for (k,parity), images_block in input_layer.items():
        for target_k, target_parity in target_keys:
            if (target_k, target_parity) not in conv_filters[(k,parity)]:
                continue
            
            filter_block = conv_filters[(k,parity)][(target_k, target_parity)]

            convolve_contracted_imgs = geom.convolve_contract(
                input_layer.D, 
                images_block, 
                filter_block, 
                input_layer.is_torus,
                stride, 
                padding,
                lhs_dilation, 
                rhs_dilation,
            )

            if ((target_k, target_parity) in layer): # it already has that key
                layer[(target_k, target_parity)] = convolve_contracted_imgs + layer[(target_k, target_parity)]
            else:
                layer.append(target_k, target_parity, convolve_contracted_imgs)

    return layer

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

def batch_scalar_activation(layer, activation_function):
    """
    Given a layer, apply the nonlinear activation function to each scalar image_block. If the layer has
    odd parity, then the activation should be an odd function.
    """
    out_layer = layer.empty()
    for (k,parity), image_block in layer.items():
        out_image_block = activation_function(image_block) if (k == 0) else image_block
        out_layer.append(k, parity, out_image_block)

    return out_layer

def norm_nonlinear(params, layer, scalar_activation=jax.nn.sigmoid, mold_params=False):
    """
    This nonlinearity has so far been unsuccessful.
    """
    params_idx, this_params = get_layer_params(params, mold_params, NORM_NONLINEAR)

    out_layer = layer.empty()
    for (k,parity), image_block in layer.items():
        norm_img = geom.norm(2 + layer.D, image_block, keepdims=True)
        in_c = image_block.shape[1]

        if mold_params:
            this_params[(k,parity)] = { 'bias': jnp.ones((1,in_c) + (1,)*layer.D + (1,)*k) }

        out_layer.append(
            k, 
            parity, 
            scalar_activation(norm_img + this_params[(k,parity)]['bias'])*image_block,
        )

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

@functools.partial(jax.jit, static_argnums=[2,3,4,5])
def VN_nonlinear(params, layer, depth=None, scalar_activation=jax.nn.relu, eps=1e-5, mold_params=False):
    """
    The vector nonlinearity in the Vector Neurons paper: https://arxiv.org/pdf/2104.12229.pdf
    Basically use the channels of a vector to get a direction vector and a value vector. Use the 
    direction vector the split the value vector space in two. In one hemisphere, the value vector
    is unchanged, while in the other it is projected to the hemispheres boundary.
    args:
        params (): params tree
        layer (BatchLayer): the input layer, must
        depth (int): out depth, defaults to None which will be set to the number of input channels
        scalar_activation (func): nonlinearity used for scalar
        eps (float): small value to avoid dividing by zero if the k_vec is close to 0, defaults to 1e-5
    """
    params_idx, this_params = get_layer_params(params, mold_params, VN_NONLINEAR)

    out_layer = layer.empty()
    for (k,parity), img_block in layer.items():
        assert k == 0 or k == 1, 'batch_VN_nonlinear: Layer can only have scalars and vectors'
        in_c = img_block.shape[1]
        out_c = in_c if depth is None else depth

        if k == 0:
            out_layer.append(k, parity, scalar_activation(img_block))
        else: # k==1
            if mold_params:
                this_params['W'] = jnp.ones((out_c,1,in_c) + (1,)*layer.D + (1,))
                this_params['U'] = jnp.ones((out_c,1,in_c) + (1,)*layer.D + (1,))

            # (batch,out_c,spatial,tensor)
            q = jnp.moveaxis(jnp.sum(this_params['W'] * img_block[None], axis=2), 0, 1)
            k_vec = jnp.moveaxis(jnp.sum(this_params['U'] * img_block[None], axis=2), 0, 1)
            k_normed = k_vec / (geom.norm(layer.D+2, k_vec, keepdims=True) + eps)

            q_k_inner_product = geom.multicontract(geom.mul(layer.D, q, k_normed, 2, 2), ((0,1),), layer.D + 2)[...,None]
            projected_q = q - q_k_inner_product * k_normed
            res = jnp.where(q_k_inner_product >= 0, q, projected_q)
            out_layer.append(k, parity, res)

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

def vector_dots_nonlinear(params, layer, depth=None, scalar_activation=jax.nn.relu, mold_params=False):
    """
    The vector nonlinearity based on scalars are universal: https://arxiv.org/abs/2106.06610
    We get a map from in_c vectors to depth vectors by taking the dot products of the channel of vectors,
    then do a little MLP on those scalars and use them as coefficients of the output vectors.
    args:
        params (): params tree
        layer (BatchLayer): the input layer, must
        depth (int): out depth, defaults to None which will be set to the number of input channels
        scalar_activation (func): nonlinearity used for scalar
    """
    params_idx, this_params = get_layer_params(params, mold_params, VECTOR_DOTS_NONLINEAR)

    out_layer = layer.empty()
    for (k,parity), img_block in layer.items():
        assert k == 0 or k == 1, 'vector_dots_nonlinear: Layer can only have scalars and vectors'
        batch,in_c = img_block.shape[:2]
        spatial,_ = geom.parse_shape(img_block.shape[2:], layer.D)
        out_c = in_c if depth is None else depth

        if k == 0:
            out_layer.append(k, parity, scalar_activation(img_block))
        else: # k==1
            if mold_params:
                this_params[(k,parity)] = {
                    'W1': jnp.ones((in_c**2,in_c*out_c)),
                    'b1': jnp.ones((1,in_c*out_c)),
                    'W2': jnp.ones((in_c*out_c,in_c*out_c)),
                    'b2': jnp.ones((1,in_c*out_c)),
                }

            # (batch,in_c,spatial,tensor) -> (batch*spatial,in_c,tensor)
            vecs = jnp.moveaxis(img_block, 1, 2+layer.D).reshape((-1,in_c) + (layer.D,)) 
            scalars = jax.vmap(lambda A: A @ A.T)(vecs) # (batch*spatial,in_c,in_c)
            scalars = scalars.reshape((len(scalars),-1)) # (batch*spatial,in_c^2)

            scalars = scalars @ this_params[(k,parity)]['W1'] + this_params[(k,parity)]['b1'] # (batch*spatial,in_c*out_c)
            scalars = scalar_activation(scalars)
            scalars = scalars @ this_params[(k,parity)]['W2'] + this_params[(k,parity)]['b2'] # (batch*spatial,in_c*out_c)

            out_vecs = jax.vmap(lambda V,S: S @ V)(vecs,scalars.reshape((len(scalars),out_c,in_c)))
            out_vecs = out_vecs.reshape((batch,) + spatial + (out_c,layer.D))

            out_layer.append(k, parity, jnp.moveaxis(out_vecs, 1+layer.D, 1))

    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

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
    Batch norm, this is not equivariant.
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
        num_channels = image_block.shape[1]
        _, k = geom.parse_shape(image_block.shape[2:], batch_layer.D)
        shape = (1,num_channels) + (1,)*batch_layer.D + (1,)*k
        if mold_params:
            this_params[key] = { SCALE: jnp.ones(shape), BIAS: jnp.ones(shape) }

            # some placeholder values for mean and variance. While mold_params=True, we are not 
            # inside pmap, so we want to avoid calling pmean
            mean = jnp.zeros(shape)
            var = jnp.ones(shape)
        elif (train):
            # both are shape (channels, (N,)*D, (D,)*k)
            mean = jax.lax.pmean(
                jnp.mean(image_block, axis=(0,) + tuple(range(2,2+batch_layer.D+k)), keepdims=True), 
                axis_name='batch',
            )
            var = jax.lax.pmean(
                jnp.mean((image_block - mean)**2, axis=(0,) + tuple(range(2,2+batch_layer.D+k)), keepdims=True),
                axis_name='batch',
            )
            assert mean.shape == var.shape == shape

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
        added_images = centered_scaled_image * this_params[key][SCALE] + this_params[key][BIAS]
        out_layer.append(key[0], key[1], added_images)
    
    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params, running_mean, running_var

def _group_norm_K1(D, image_block, groups, method='eigh', eps=1e-5):
    """
    Perform the layer norm whitening on a vector image block. This is somewhat based on the Clifford
    Layers Batch norm, link below. However, this differs in that we use eigh rather than cholesky because
    cholesky is not invariant to all the elements of our group.
    https://github.com/microsoft/cliffordlayers/blob/main/cliffordlayers/nn/functional/batchnorm.py

    args:
        D (int): the dimension of the space
        image_block (jnp.array): data block of shape (batch,channels,spatial,tensor)
        groups (int): the number of channel groups, must evenly divide channels
        method (string): method used for the whitening, either 'eigh', or 'cholesky'. Note that
            'cholesky' is not equivariant.
        eps (float): to avoid non-invertible matrices, added to the covariance matrix
    """
    batch,in_c = image_block.shape[:2]
    spatial_dims, k = geom.parse_shape(image_block.shape[2:], D)
    assert k == 1, f'ml::_group_norm_K1: Equivariant group_norm is not implemented for k>1, but k={k}'
    assert (in_c % groups) == 0 # groups must evenly divide the number of channels
    channels_per_group = in_c // groups

    image_grouped = image_block.reshape((batch, groups, channels_per_group) + spatial_dims + (D,))

    mean = jnp.mean(image_grouped, axis=tuple(range(2,3+D)), keepdims=True) #(B,G,1,(1,)*D,D)
    centered_img = image_grouped - mean # (B,G,in_c//G,spatial,tensor)

    X = centered_img.reshape((batch,groups,-1,D)) # (B,G,spatial*in_c//G,D)
    cov = jnp.einsum('...ij,...ik->...jk', X, X) / X.shape[-2] # biased cov, (B,G,D,D)

    if method == 'eigh':
        # symmetrize_input=True seems to cause issues with autograd, and cov is already symmetric
        eigvals, eigvecs = jnp.linalg.eigh(cov, symmetrize_input=False)
        eigvals_invhalf = jnp.sqrt(1./(eigvals+eps))
        S_diag = jax.vmap(lambda S: jnp.diag(S))(eigvals_invhalf.reshape((-1,D))).reshape((batch,groups,D,D))
        # do U S U^T, and multiply each vector in centered_img by the resulting matrix
        whitened_data = jnp.einsum(
            '...ij,...jk,...kl,...ml->...mi',
            eigvecs,
            S_diag,
            eigvecs.transpose((0,1,3,2)),
            centered_img.reshape((batch,groups,-1,D)),
        )
    elif method == 'cholesky':
        L = jax.lax.linalg.cholesky(cov, symmetrize_input=False) # (batch*groups,D,D)
        L = L + eps*jnp.eye(D).reshape((1,D,D,))
        whitened_data = jax.lax.linalg.triangular_solve(
            L, 
            centered_img.reshape((batch*groups,-1) + (D,)),
            left_side=False,
            lower=True,
        )
    else:
        raise NotImplementedError(f'ml::_group_norm_K1: method {method} not implemented.')
    
    return whitened_data.reshape(image_block.shape)

functools.partial(jax.jit, static_argnums=[2,3,4])
def group_norm(params, layer, groups, eps=1e-5, equivariant=True, mold_params=False):
    """
    Implementation of group_norm. When num_groups=num_channels, this is equivalent to instance_norm. When
    num_groups=1, this is equivalent to layer_norm. This function takes in a BatchLayer, not a Layer.
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input layer, each image is shape (batch,in_c,spatial)
        groups (int): the number of channel groups for group_norm
        eps (float): number to add to variance so we aren't dividing by 0
        equivariant (bool): defaults to True
        mold_params (bool): parameter to control whether to shape the parameters.
    """
    params_idx, this_params = get_layer_params(params, mold_params, GROUP_NORM)
    if mold_params:
        this_params = { SCALE: {} }

    out_layer = layer.empty()
    for (k,parity), image_block in layer.items():
        batch,in_c = image_block.shape[:2]
        spatial_dims, _ = geom.parse_shape(image_block.shape[2:], layer.D)
        assert (in_c % groups) == 0 # groups must evenly divide the number of channels
        channels_per_group = in_c // groups

        image_grouped = image_block.reshape((batch, groups, channels_per_group) + spatial_dims + (layer.D,)*k)

        if equivariant and k==1:
            whitened_data = _group_norm_K1(layer.D, image_block, groups, eps=eps)
        elif equivariant and k>1:
            raise NotImplementedError(
                f'ml::group_norm: Currently equivariant group_norm is not implemented for k>1, but k={k}',
            )
        else:
            mean = jnp.mean(image_grouped, axis=tuple(range(2,3+layer.D+k)), keepdims=True)
            var = jnp.var(image_grouped, axis=tuple(range(2,3+layer.D+k)), keepdims=True)
            assert mean.shape == var.shape == (batch,groups,1) + (1,)*layer.D + (1,)*k
            whitened_data = ((image_grouped - mean)/jnp.sqrt(var + eps)).reshape(image_block.shape)

        if mold_params:
            this_params[SCALE][(k,parity)] = jnp.ones((1,in_c) + (1,)*layer.D + (1,)*k)

        scaled_image = whitened_data * this_params[SCALE][(k,parity)]
        out_layer.append(k, parity, scaled_image)

    out_layer, this_params = add_bias(this_params, out_layer, mean_bias=equivariant, mold_params=mold_params)
    
    params = update_params(params, params_idx, this_params, mold_params)

    return out_layer, params

@functools.partial(jit, static_argnums=[2,3])
def layer_norm(params, input_layer, eps=1e-05, mold_params=False):
    """
    Implementation of layer norm based on group_norm.
    """
    return group_norm(params, input_layer, 1, eps, mold_params)

@functools.partial(jax.jit, static_argnums=[1,2])
def batch_max_pool(input_layer, patch_len, use_norm=True):
    """
    Max pool layer on a BatchLayer. Patch len must divide evenly every spatial_dim, and use_norm can
    only be used with (pseudo-)scalars, and it breaks equivariance for pseudoscalars.
    args:
        input_layer (BatchLayer): input data
        patch_len (int): side of the max pool sides
        use_norm (bool): whether to use the Frobenius norm on the tensor, defaults to True
    """
    vmap_max_pool = jax.vmap(
        jax.vmap(geom.max_pool, in_axes=(None, 0, None, None)), 
        in_axes=(None, 0, None, None),
    )

    out_layer = input_layer.empty()
    for (k,parity), image_block in input_layer.items():
        out_layer.append(k, parity, vmap_max_pool(input_layer.D, image_block, patch_len, use_norm))

    return out_layer

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
    with jax.disable_jit(): # this could be slow, lets see?
        params = net_func(defaultdict(lambda: None), input_layer, subkey, True, return_params=True)[-1]

    initializers = {
        BATCH_NORM: batch_norm_init,
        GROUP_NORM: group_norm_init,
        CHANNEL_COLLAPSE: channel_collapse_init,
        CONV: functools.partial(conv_init, D=input_layer.D),
        CONV_OLD: conv_old_init,
        CASCADING_CONTRACTIONS: cascading_contractions_init,
        PARAMED_CONTRACTIONS: paramed_contractions_init,
        EQUIV_DENSE: equiv_dense_init,
        VN_NONLINEAR: VN_nonlinear_init,
        NORM_NONLINEAR: norm_nonlinear_init,
        VECTOR_DOTS_NONLINEAR: vector_dots_init,
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
        out_params[key] = { SCALE: jnp.ones(inner_tree[SCALE].shape) }
        # bias violates equivariance for certain tensor/parities, so it might be missing
        if BIAS in inner_tree: 
            out_params[key][BIAS] = jnp.zeros(inner_tree[BIAS].shape)

    return out_params

def group_norm_init(rand_key, tree):
    out_params = {}
    if BIAS in tree:
        out_params[BIAS] = {}
        for key, params_block in tree[BIAS].items():
            out_params[BIAS][key] = jnp.zeros(params_block.shape)

    if SCALE in tree:
        out_params[SCALE] = {}
        for key, params_block in tree[SCALE].items():
            out_params[SCALE][key] = jnp.ones(params_block.shape)

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
        for filter_key, (params_shape, filter_block_shape) in d.items():
            rand_key, subkey = random.split(rand_key)
            bound = 1/jnp.sqrt(np.multiply.reduce(filter_block_shape))
            params[key][filter_key] = random.uniform(subkey, shape=params_shape, minval=-bound, maxval=bound)

    out_params[filter_type] = params

    if (BIAS in tree):
        bias_params = {}
        for key, params_block in tree[BIAS].items():
            # reuse the bound from above, it shouldn't be any different
            rand_key, subkey = random.split(rand_key)
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

def VN_nonlinear_init(rand_key, tree):
    out_params = {}
    for key, param_block in tree.items():
        rand_key, subkey = random.split(rand_key)
        bound = 1./jnp.sqrt(param_block.shape[2])
        out_params[key] = random.uniform(subkey, shape=param_block.shape, minval=-bound, maxval=bound)

    return out_params

def norm_nonlinear_init(rand_key, tree):
    out_params = {}
    for key, param_block_dict in tree.items():
        param_block = param_block_dict['bias']
        rand_key, subkey = random.split(rand_key)
        bound = 1./jnp.sqrt(param_block.shape[1])
        out_params[key] = { 'bias': random.uniform(subkey, shape=param_block.shape, minval=-bound, maxval=bound) }

    return out_params

def vector_dots_init(rand_key, tree):
    out_params = {}
    for key, param_block_dict in tree.items():
        out_params[key] = {}
        rand_key, subkey = random.split(rand_key)
        bound = 1./jnp.sqrt(len(param_block_dict['W1']))
        out_params[key]['W1'] = random.uniform(subkey, shape=param_block_dict['W1'].shape, minval=-bound, maxval=bound)
        out_params[key]['b1'] = random.uniform(subkey, shape=param_block_dict['b1'].shape, minval=-bound, maxval=bound)

        bound = 1./jnp.sqrt(len(param_block_dict['W2']))
        out_params[key]['W2'] = random.uniform(subkey, shape=param_block_dict['W2'].shape, minval=-bound, maxval=bound)
        out_params[key]['b2'] = random.uniform(subkey, shape=param_block_dict['b2'].shape, minval=-bound, maxval=bound)

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

def smse_loss(layer_x, layer_y):
    """
    Sum of mean squared error loss. The sum is over the channels, the mean is over the spatial dimensions and
    the batch.
    """
    spatial_size = np.multiply.reduce(layer_x.get_spatial_dims())
    return jnp.mean(jnp.sum((layer_x.to_vector() - layer_y.to_vector())**2/spatial_size, axis=1))

def l2_loss(x, y):
    return jnp.sqrt(l2_squared_loss(x, y))

def l2_squared_loss(x, y):
    return jnp.sum((x - y) ** 2)

def pointwise_normalized_loss(layer_x, layer_y, eps=1e-5):
    """
    Pointwise normalized loss. We find the norm of each channel at each spatial point of the true value
    and divide the tensor by that norm. Then we take the l2 loss, mean over the spatial dimensions, sum 
    over the channels, take a final square root and mean over the batch.
    args:
        layer_x (BatchLayer): input batch layer
        layer_y (BatchLayer): target batch layer
    """
    spatial_size = np.multiply.reduce(layer_x.get_spatial_dims())

    order_loss = jnp.zeros(layer_x.get_L())
    for (k,parity), img_block in layer_y.items():
        norm = geom.norm(layer_y.D + 2, img_block, keepdims=True)
        normalized_l2 = ((layer_x[(k,parity)] - img_block)**2) / (norm + eps)
        order_loss = order_loss + (jnp.sum(normalized_l2, axis=range(1,img_block.ndim)) / spatial_size)

    return jnp.mean(jnp.sqrt(order_loss))

## Data and Batching operations

def get_batch_layer(layers, batch_size, rand_key, devices=None):
    """
    Given a set of layers, construct random batches of those layers. The most common use case is for
    layers to be a tuple (X,Y) so that the batches have the inputs and outputs. In this case, it will return
    a list of length 2 where the first element is a list of the batches of the input data and the second
    element is the same batches of the output data. Automatically reshapes the batches to use with 
    pmap based on the number of gpus found. 
    args:
        layers (BatchLayer or iterable of BatchLayer): batch layers which all get simultaneously batched
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness. If None, the order won't be random
        devices (list): gpu/cpu devices to use, if None (default) then sets this to jax.devices()
    returns: list of lists of batches (which are BatchLayers)
    """
    if isinstance(layers, geom.BatchLayer):
        layers = (layers,)

    L = layers[0].get_L()
    batch_indices = jnp.arange(L) if rand_key is None else random.permutation(rand_key, L)

    if devices is None:
        devices = jax.devices()

    batches = [[] for _ in range(len(layers))]
    # if L is not divisible by batch, the remainder will be ignored
    for i in range(int(math.floor(L / batch_size))): #iterate through the batches of an epoch
        idxs = batch_indices[i*batch_size:(i+1)*batch_size]
        for j, layer in enumerate(layers):
            batches[j].append(layer.get_subset(idxs).reshape_pmap(devices))

    return batches if (len(batches) > 1) else batches[0]

def map_loss_in_batches(
    map_and_loss, 
    params, 
    layer_X, 
    layer_Y, 
    batch_size, 
    rand_key, 
    train, 
    has_aux=False, 
    aux_data=None,
    devices=None,
):
    """
    Runs map_and_loss for the entire layer_X, layer_Y, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number 
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_and_loss (function): function that takes in params, X_batch, Y_batch, rand_key, train, and 
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        params (params tree): the params to run through map_and_loss
        layer_X (BatchLayer): input data
        layer_Y (BatchLayer): target output data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        train (bool): whether this is training or not, likely not
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        devices (list): gpu/cpu devices to use
    returns: average loss over the entire layer
    """
    if has_aux:
        pmap_loss_grad = jax.pmap(
            map_and_loss, 
            axis_name='batch', 
            in_axes=(None, 0, 0, None, None, None),
            out_axes=(0, None),
            static_broadcasted_argnums=4,
            devices=devices,
        )
    else:
        pmap_loss_grad = jax.pmap(
            map_and_loss, 
            axis_name='batch', 
            in_axes=(None, 0, 0, None, None),
            static_broadcasted_argnums=4,
            devices=devices,
        )

    rand_key, subkey = random.split(rand_key)
    X_batches, Y_batches = get_batch_layer((layer_X, layer_Y), batch_size, subkey, devices)
    total_loss = 0
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        rand_key, subkey = random.split(rand_key)
        if (has_aux):
            one_loss, aux_data = pmap_loss_grad(params, X_batch, Y_batch, subkey, train, aux_data)
        else:
            one_loss = pmap_loss_grad(params, X_batch, Y_batch, subkey, False)

        total_loss += jnp.mean(one_loss)

    return total_loss / len(X_batches)

def map_in_batches(
    map_f, 
    params, 
    layer_X, 
    batch_size, 
    rand_key, 
    train, 
    has_aux=False, 
    aux_data=None,
    devices=None,
):
    """
    Runs map_f for the entire layer_X, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map_f when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number 
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_f (function): function that takes in params, X_batch, rand_key, train, and 
            aux_data if has_aux is true, and returns the mapped layer, and aux_data if has_aux is true.
        params (params tree): the params to run through map_f
        layer_X (BatchLayer): input data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        train (bool): whether this is training or not, likely not
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        devices (list): gpu/cpu devices to use
    returns: average loss over the entire layer
    """
    if has_aux:
        pmap_f = jax.pmap(
            map_f, 
            axis_name='batch', 
            in_axes=(None, 0, None, None, None),
            out_axes=(0, None),
            static_broadcasted_argnums=3,
            devices=devices,
        )
    else:
        pmap_f = jax.pmap(map_f, 
            axis_name='batch', 
            in_axes=(None, 0, None, None),
            static_broadcasted_argnums=3,
            devices=devices,
        )

    results = []
    for X_batch in get_batch_layer(layer_X, batch_size, None, devices):
        rand_key, subkey = random.split(rand_key)
        if (has_aux):
            batch_out, aux_data = pmap_f(params, X_batch, subkey, train, aux_data)
        else:
            batch_out = pmap_f(params, X_batch, subkey, False)

        results.append(batch_out)

    return results

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

    def stop(self, params, current_epoch, train_loss, val_loss, batch_time):
        pass

    def log_status(self, epoch, train_loss, val_loss, batch_time):
        if (train_loss is not None):
            if (val_loss is not None):
                print(
                    f'Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Batch time: {batch_time:.5f}',
                )
            else:
                print(f'Epoch {epoch} Train: {train_loss:.7f} Batch time: {batch_time:.5f}')

class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self, epochs, verbose=0) -> None:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(self, params, current_epoch, train_loss, val_loss, batch_time) -> bool:
        self.best_params = params

        if (
            self.verbose == 2 or
            (self.verbose == 1 and (current_epoch % (self.epochs // np.min([10,self.epochs])) == 0))
        ):
            self.log_status(current_epoch, train_loss, val_loss, batch_time)

        return current_epoch >= self.epochs
    
class TrainLoss(StopCondition):
    # Stop when the training error stops improving after patience number of epochs.

    def __init__(self, patience=0, min_delta=0, verbose=0) -> None:
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(self, params, current_epoch, train_loss, val_loss, batch_time) -> bool:
        if (train_loss is None):
            return False
        
        if (train_loss < (self.best_train_loss - self.min_delta)):
            self.best_train_loss = train_loss
            self.best_params = params
            self.epochs_since_best = 0

            if (self.verbose >= 1):
                self.log_status(current_epoch, train_loss, val_loss, batch_time)
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

    def stop(self, params, current_epoch, train_loss, val_loss, batch_time) -> bool:
        if (val_loss is None):
            return False
        
        if (val_loss < (self.best_val_loss - self.min_delta)):
            self.best_val_loss = val_loss
            self.best_params = params
            self.epochs_since_best = 0

            if (self.verbose >= 1):
                self.log_status(current_epoch, train_loss, val_loss, batch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience

@jit
def grads_mean(grads):
    """
    Recursively take the mean over the first axis of every jnp.array in the tree of gradients.
    args:
        grads (tree of jnp.arrays): the grads
    """
    # mean over a grads dictionary
    if not isinstance(grads, dict): # is a jnp.ndarray
        return jnp.mean(grads, axis=0)
    
    out_grads = {}
    for k,v in grads.items():
        out_grads[k] = grads_mean(v)

    return out_grads

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
    devices=None,
):
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The params are returned. This function automatically
    pmaps over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    In order to do collectives across the pmap batches, use axis_name='batch', such as 
    jnp.pmean(x, axis_name='batch').
    args:
        X (BatchLayer): The X input data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        Y (BatchLayer): The Y target data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss (function): function that takes in params, X_batch, Y_batch, rand_key, and train and
            returns the loss. If has_aux is True, then it also takes in aux_data and returns aux_data.
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
        devices (list): gpu/cpu devices to use, if None (default) then it will use jax.devices()
    """
    if (isinstance(stop_condition, ValLoss)):
        assert validation_X and validation_Y

    if checkpoint_kwargs is None:
        batch_loss_grad = value_and_grad(map_and_loss, has_aux=has_aux)
    else:
        batch_loss_grad = checkpoint(value_and_grad(map_and_loss, has_aux=has_aux), **checkpoint_kwargs)

    if has_aux:
        pmap_loss_grad = jax.pmap(
            batch_loss_grad, 
            axis_name='batch', 
            in_axes=(None, 0, 0, None, None, None),
            out_axes=((0, None), 0),
            static_broadcasted_argnums=4,
            devices=devices,
        )
    else:
        pmap_loss_grad = jax.pmap(
            batch_loss_grad, 
            axis_name='batch', 
            in_axes=(None, 0, 0, None, None),
            static_broadcasted_argnums=4,
            devices=devices,
        )

    if (optimizer is None):
        optimizer = optax.adam(0.1)

    opt_state = optimizer.init(params)

    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    train_loss = []
    val_loss = []
    batch_times = []
    while (
        not stop_condition.stop(params, epoch, epoch_loss, epoch_val_loss, jnp.mean(jnp.array(batch_times)))
    ):
        if noise_stdev:
            rand_key, subkey = random.split(rand_key)
            train_X = add_noise(X, noise_stdev, subkey)
        else:
            train_X = X

        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batch_layer((train_X, Y), batch_size, subkey, devices)
        epoch_loss = 0
        batch_times = []
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            rand_key, subkey = random.split(rand_key)
            start_time = time.time()
            if (has_aux):
                (pmap_loss_val, aux_data), pmap_grads = pmap_loss_grad(
                    params, 
                    X_batch, 
                    Y_batch, 
                    subkey, 
                    True,
                    aux_data,
                )
            else:
                pmap_loss_val, pmap_grads = pmap_loss_grad(params, X_batch, Y_batch, subkey, True)

            batch_times.append(time.time() - start_time)
            updates, opt_state = optimizer.update(grads_mean(pmap_grads), opt_state, params)
            params = optax.apply_updates(params, updates)
            epoch_loss += jnp.mean(pmap_loss_val)

        epoch_loss = epoch_loss / len(X_batches)
        train_loss.append(epoch_loss)

        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if (validation_X and validation_Y):
            rand_key, subkey = random.split(rand_key)
            epoch_val_loss = map_loss_in_batches(
                map_and_loss, 
                params, 
                validation_X, 
                validation_Y, 
                batch_size, 
                subkey, 
                False, # train
                has_aux,
                aux_data,
                devices,
            )
            val_loss.append(epoch_val_loss)

        if (save_params and ((epoch % 10) == 0)):
            jnp.save(save_params, stop_condition.best_params)

    if (has_aux):
        return stop_condition.best_params, aux_data, jnp.array(train_loss), jnp.array(val_loss)
    else:
        return stop_condition.best_params, jnp.array(train_loss), jnp.array(val_loss)

BENCHMARK_DATA = 'benchmark_data'
BENCHMARK_MODEL = 'benchmark_model'
BENCHMARK_NONE = 'benchmark_none'

def benchmark(
    get_data,
    models,
    rand_key, 
    benchmark,
    benchmark_range,
    benchmark_type=BENCHMARK_DATA,
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
        benchmark_type (str): one of { BENCHMARK_DATA, BENCHMARK_MODEL, BENCHMARK_NONE }, says 
        num_trials (int): number of trials to run, defaults to 1
        num_results (int): the number of results that will come out of the model function. If num_results is
            greater than 1, it should be indexed by range(num_results)
    returns:
        an np.array of shape (trials, benchmark_range, models, num_results) with the results all filled in
    """
    assert benchmark_type in { BENCHMARK_DATA, BENCHMARK_MODEL, BENCHMARK_NONE }
    if benchmark_type == BENCHMARK_NONE:
        benchmark = ''
        benchmark_range = [0]

    results = np.zeros((num_trials, len(benchmark_range), len(models), num_results))
    for i in range(num_trials):
        for j, benchmark_val in enumerate(benchmark_range):

            data_kwargs = { benchmark: benchmark_val } if benchmark_type == BENCHMARK_DATA else {}
            model_kwargs = { benchmark: benchmark_val } if benchmark_type == BENCHMARK_MODEL else {}

            rand_key, subkey = random.split(rand_key)
            data = get_data(subkey, **data_kwargs)

            for k, (model_name, model) in enumerate(models):
                print(f'trial {i} {benchmark}: {benchmark_val:.4f} {model_name}')

                rand_key, subkey = random.split(rand_key)
                res = model(
                    data, 
                    subkey, 
                    f'{model_name}_{benchmark}{benchmark_val}_t{i}',
                    **model_kwargs,
                )

                if num_results > 1:
                    for q in range(num_results):
                        results[i,j,k,q] = res[q]
                else:
                    results[i,j,k,0] = res

    return results
