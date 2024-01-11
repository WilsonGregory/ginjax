import functools
from collections import defaultdict

import jax.numpy as jnp
import jax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def unet_conv_block(
    params, 
    layer, 
    train, 
    num_conv, 
    conv_args, 
    conv_kwargs, 
    batch_stats=None, 
    batch_stats_idx=None, 
    activation_f=None,
    equivariant=False,
    mold_params=False,
):
    conv_f = ml.batch_conv_contract if equivariant else ml.batch_conv_layer
    for _ in range(num_conv):
        layer, params = conv_f(
            params, 
            layer,
            *conv_args,
            **conv_kwargs,
            mold_params=mold_params,
        )
        if batch_stats is not None:
            layer, params, mean, var = ml.batch_norm(
                params, 
                layer, 
                train, 
                batch_stats[batch_stats_idx]['mean'],
                batch_stats[batch_stats_idx]['var'],
                mold_params=mold_params,
            )
            batch_stats[batch_stats_idx] = { 'mean': mean, 'var': var }
            batch_stats_idx += 1
        if activation_f is not None:
            layer = ml.batch_scalar_activation(layer, activation_f)

    return layer, params, batch_stats, batch_stats_idx

@functools.partial(jax.jit, static_argnums=[3,5,6,7])
def unet2015(params, layer, key, train, batch_stats=None, depth=64, activation_f=jax.nn.gelu, return_params=False):
    num_downsamples = 4
    num_conv = 2

    # convert to channels of a scalar layer
    layer = layer.to_scalar_layer()

    batch_stats_idx = 0
    if batch_stats is None:
        batch_stats = defaultdict(lambda: { 'mean': None, 'var': None })

    # first we do the downsampling
    residual_layers = []
    for downsample in range(num_downsamples):
        layer, params, batch_stats, batch_stats_idx = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            [{ 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } }], # conv_args
            { 'depth': depth*(2**downsample) }, # conv_kwargs
            batch_stats,
            batch_stats_idx,
            activation_f,
            mold_params=return_params,
        )
        residual_layers.append(layer)
        layer = ml.batch_max_pool(layer, 2)

    # bottleneck layer
    layer, params, batch_stats, batch_stats_idx = unet_conv_block(
        params, 
        layer, 
        train, 
        num_conv, 
        [{ 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } }], # conv_args
        { 'depth': depth*(2**num_downsamples) }, # conv_kwargs
        batch_stats,
        batch_stats_idx,
        activation_f,
        mold_params=return_params,
    )

    # now we do the upsampling and concatenation
    for upsample in reversed(range(num_downsamples)):

        # upsample
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            { 'type': 'free', 'M': 2, 'filter_key_set': { (0,0) } },
            depth*(2**upsample),
            padding=((1,1),)*layer.D,
            lhs_dilation=(2,)*layer.D, # do the transposed convolution
            mold_params=return_params,
        )
        # concat the upsampled layer and the residual
        layer = jax.vmap(lambda layer1, layer2: layer1.concat(layer2))(layer, residual_layers[upsample])

        layer, params, batch_stats, batch_stats_idx = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            [{ 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } }], # conv_args
            { 'depth': depth*(2**upsample) }, # conv_kwargs
            batch_stats,
            batch_stats_idx,
            activation_f,
            mold_params=return_params,
        )

    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
        3, # number of output channels, one scalar and 2 vector
        mold_params=return_params,
    )

    # swap the vector back to the vector img
    layer = geom.BatchLayer(
        {
            (0,0): jnp.expand_dims(layer[(0,0)][:,0], axis=1),
            (1,0): jnp.expand_dims(layer[(0,0)][:,1:].transpose((0,2,3,1)), axis=1),
        },
        layer.D,
        layer.is_torus,
    )

    return (layer, batch_stats, params) if return_params else (layer, batch_stats)

@functools.partial(jax.jit, static_argnums=[3,6,7,8])
def unet2015_equiv(
    params, 
    layer, 
    key, 
    train, 
    conv_filters, 
    upsample_filters, # 2x2 filters
    depth=64, 
    activation_f=jax.nn.gelu, 
    return_params=False,
):
    num_downsamples = 4
    num_conv = 2

    # first we do the downsampling
    residual_layers = []
    for downsample in range(num_downsamples):
        layer, params, _, _ = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            [conv_filters], # conv_args
            { 'depth': depth*(2**downsample), 'target_keys': ((0,0),(1,0)) }, # conv_kwargs
            activation_f=activation_f,
            equivariant=True,
            mold_params=return_params,
        )
        residual_layers.append(layer)
        layer = ml.batch_max_pool(layer, 2)

    # bottleneck layer
    layer, params, _, _ = unet_conv_block(
        params, 
        layer, 
        train, 
        num_conv, 
        [conv_filters], # conv_args
        { 'depth': depth*(2**num_downsamples), 'target_keys': ((0,0),(1,0)) }, # conv_kwargs
        activation_f=activation_f,
        equivariant=True,
        mold_params=return_params,
    )

    # now we do the upsampling and concatenation
    for upsample in reversed(range(num_downsamples)):

        # upsample
        layer, params = ml.batch_conv_contract(
            params, 
            layer,
            upsample_filters,
            depth*(2**upsample),
            ((0,0),(1,0)),
            padding=((1,1),)*layer.D,
            lhs_dilation=(2,)*layer.D, # do the transposed convolution
            mold_params=return_params,
        )
        # concat the upsampled layer and the residual
        layer = jax.vmap(lambda layer1, layer2: layer1.concat(layer2))(layer, residual_layers[upsample])

        layer, params, _, _ = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            [conv_filters], # conv_args
            { 'depth': depth*(2**upsample), 'target_keys': ((0,0),(1,0)) }, # conv_kwargs
            activation_f=activation_f,
            equivariant=True,
            mold_params=return_params,
        )

    layer, params = ml.batch_conv_contract(
        params, 
        layer,
        conv_filters,
        1,
        ((0,0),(1,0)),
        mold_params=return_params,
    )

    return (layer, params) if return_params else layer

@functools.partial(jax.jit, static_argnums=[3,4,5,6,8])
def dil_resnet(
    params, 
    layer, 
    key, 
    train, 
    depth=48, 
    activation_f=jax.nn.relu, 
    equivariant=False, 
    conv_filters=None, 
    return_params=False,
):
    assert layer.D == 2
    num_blocks = 4

    if equivariant:
        assert conv_filters is not None
        conv_f = ml.batch_conv_contract
        conv_args = [conv_filters, depth, ((0,0),(1,0))]
    else:
        layer = layer.to_scalar_layer()
        conv_f = ml.batch_conv_layer
        conv_args = [{ 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } }, depth]

    # encoder
    layer, params = conv_f(
        params, 
        layer,
        *conv_args,
        bias=False if equivariant else True,
        mold_params=return_params,
    )

    for _ in range(num_blocks):

        residual_layer = layer.copy()
        # dCNN block

        for dilation in [1,2,4,8,4,2,1]:
            layer, params = conv_f(
                params, 
                layer,
                *conv_args,
                bias=False if equivariant else True,
                mold_params=return_params,
                rhs_dilation=(dilation,)*layer.D,
            )
            if activation_f is not None:
                layer = ml.batch_scalar_activation(layer, activation_f)

        layer = geom.BatchLayer.from_vector(layer.to_vector() + residual_layer.to_vector(), layer)

    # decoder
    if equivariant:
        layer, params = conv_f(
            params,
            layer,
            conv_filters,
            1,
            ((0,0),(1,0)),
            mold_params=return_params,
        )
    else:
        layer, params = conv_f(
            params, 
            layer,
            { 'type': 'free', 'M': 3, 'filter_key_set': { (0,0) } },
            depth=3, 
            bias=True,
            mold_params=return_params,
        )

        # swap the vector back to the vector img
        layer = layer.from_scalar_layer({ (0,0): 1, (1,0): 1 })

    return (layer, params) if return_params else layer