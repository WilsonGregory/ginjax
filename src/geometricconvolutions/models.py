import functools
from collections import defaultdict
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

def unet_conv_block(
    params, 
    layer, 
    train, 
    num_conv, 
    conv_kwargs, 
    batch_stats=None, 
    batch_stats_idx=None, 
    activation_f=None,
    mold_params=False,
):
    """
    The Conv block for the U-Net, include batch norm and an activation function.
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        train (bool): whether train mode or test mode, relevant for batch_norm
        num_conv (int): the number of convolutions, how many times to repeat the block
        conv_kwargs (dict): keyword args to pass the the convolution function
        batch_stats (dict): state for batch_norm, also determines whether batch_norm layer happens
        batch_stats_idx (int): the index of batch_norm that we are on
        activation_f (string or function): the function that we pass to batch_scalar_activation
        mold_params (bool): whether we are mapping the params, or applying them
    returns: layer, params, batch_stats, batch_stats_idx 
    """
    for _ in range(num_conv):
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
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
        layer, params = handle_activation(activation_f, params, layer, mold_params)

    return layer, params, batch_stats, batch_stats_idx

ACTIVATION_REGISTRY = {
    'relu': jax.nn.relu,
    'gelu': jax.nn.gelu,
    'tanh': jax.nn.tanh,
}

def handle_activation(activation_f, params, layer, mold_params):
    if activation_f is None:
        return layer, params
    elif isinstance(activation_f, str):
        if activation_f == ml.VN_NONLINEAR:
            return ml.VN_nonlinear(params, layer, mold_params=mold_params)
        elif activation_f == ml.NORM_NONLINEAR:
            return ml.norm_nonlinear(params, layer, mold_params=mold_params)
        else:
            return ml.batch_scalar_activation(layer, ACTIVATION_REGISTRY[activation_f]), params
    else:
        return ml.batch_scalar_activation(layer, activation_f), params

@functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8,11,12,13])
def unetBase(
    params, 
    layer, 
    key, 
    train, 
    output_keys=None,
    output_depth=1,
    depth=64, 
    activation_f=jax.nn.gelu,
    equivariant=False,
    conv_filters=None, 
    upsample_filters=None, # 2x2 filters
    use_group_norm=True,
    return_params=False,
    mold_params=None,
):
    """
    The U-Net from the original 2015 paper. The involves 4 downsampling steps and 4 upsampling steps, 
    each interspersed with 2 convolutions interleaved with scalar activations and batch norm. The downsamples
    are achieved with max_pool layers with patch length of 2 and the upsamples are achieved with transposed
    convolution. Additionally, there are residual connections after every downsample and upsample. For the 
    equivariant version, we use invariant filters, norm max_pool, and no batch_norm.
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        key (jnp.random key): key for any layers requiring randomization
        train (bool): whether train mode or test mode, relevant for batch_norm
        batch_stats (dict): state for batch_norm, default None
        output_keys (tuple of tuples of ints): the output key types of the model. For the Shallow Water
            experiments, should be ((0,0),(1,0)) for the pressure/velocity form and ((0,0),(0,1)) for 
            the pressure/vorticity form.
        output_depth (int or tuple): output depth for each output key. If an int, then the depth is 
            that int for each key. Defaults to 1, cannot be None. Needs to be hashable, so the structure
            is ( (key,depth), (key,depth)) where key itself is a tuple like (0,0)
        depth (int): the depth of the layers, defaults to 64
        activation_f (string or function): the function that we pass to batch_scalar_activation
        equivariant (bool): whether to use the equivariant version of the model, defaults to False
        conv_filters (Layer): the conv filters used for the equivariant version
        upsample_filters (Layer): the conv filters used for the upsample layer of the equivariant version
        use_group_norm (bool): whether to use group_norm, defaults to True
        return_params (bool): whether we are mapping the params, or applying them
        mold_params (bool): 
    """
    num_downsamples = 4
    num_conv = 2

    if mold_params is None:
        mold_params = return_params

    assert output_keys is not None
    assert output_depth is not None

    if isinstance(output_depth, int):
        output_depth = tuple((key,output_depth) for key in output_keys)

    if equivariant:
        assert (conv_filters is not None) and (upsample_filters is not None)
        filter_info = { 'type': ml.CONV_FIXED, 'filters': conv_filters }
        filter_info_upsample = { 'type': ml.CONV_FIXED, 'filters': upsample_filters }
        target_keys = output_keys
    else:
        filter_info = { 'type': ml.CONV_FREE, 'M': 3 }
        filter_info_upsample = { 'type': ml.CONV_FREE, 'M': 2 }
        target_keys = ((0,0),)

        # convert to channels of a scalar layer
        layer = layer.to_scalar_layer()

    # embedding
    for _ in range(num_conv):
        layer, params = ml.batch_conv_layer(
            params,
            layer,
            filter_info,
            depth,
            target_keys,
            bias=True,
            mold_params=mold_params,
        )
        if use_group_norm:
            layer, params = ml.group_norm(params, layer, 1, equivariant=equivariant, mold_params=mold_params)
        layer, params = handle_activation(activation_f, params, layer, mold_params)

    # first we do the downsampling
    residual_layers = []
    for downsample in range(1,num_downsamples+1):

        residual_layers.append(layer)

        layer = ml.batch_max_pool(layer, 2, use_norm=equivariant)

        for _ in range(num_conv):
            if use_group_norm: # try the pre-activations
                layer, params = ml.group_norm(params, layer, 1, equivariant=equivariant, mold_params=mold_params)
            layer, params = handle_activation(activation_f, params, layer, mold_params)

            layer, params = ml.batch_conv_layer(
                params,
                layer,
                filter_info,
                depth*(2**downsample),
                target_keys,
                bias=True,
                mold_params=mold_params,
            )
            

    # now we do the upsampling and concatenation
    for upsample in reversed(range(num_downsamples)):

        # upsample
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            filter_info_upsample,
            depth*(2**upsample),
            target_keys,
            padding=((1,1),)*layer.D,
            lhs_dilation=(2,)*layer.D, # do the transposed convolution
            bias=True,
            mold_params=mold_params,
        )
        # concat the upsampled layer and the residual
        layer = layer.concat(residual_layers[upsample], axis=1)

        for _ in range(num_conv):
            if use_group_norm:
                layer, params = ml.group_norm(params, layer, 1, equivariant=equivariant, mold_params=mold_params)
            layer, params = handle_activation(activation_f, params, layer, mold_params)
            
            layer, params = ml.batch_conv_layer(
                params,
                layer,
                filter_info,
                depth*(2**upsample),
                target_keys,
                bias=True,
                mold_params=mold_params,
            )
            

    layer, params = ml.batch_conv_layer(
        params,
        layer,
        filter_info,
        output_depth,
        output_keys,
        bias=True,
        mold_params=mold_params,
    )

    return (layer, params) if return_params else layer

@functools.partial(jax.jit, static_argnums=[3,5,6,7,8,11])
def unet2015(
    params, 
    layer, 
    key, 
    train, 
    batch_stats=None,
    output_keys=None,
    depth=64, 
    activation_f=jax.nn.gelu,
    equivariant=False,
    conv_filters=None, 
    upsample_filters=None, # 2x2 filters
    return_params=False,
):
    """
    The U-Net from the original 2015 paper. The involves 4 downsampling steps and 4 upsampling steps, 
    each interspersed with 2 convolutions interleaved with scalar activations and batch norm. The downsamples
    are achieved with max_pool layers with patch length of 2 and the upsamples are achieved with transposed
    convolution. Additionally, there are residual connections after every downsample and upsample. For the 
    equivariant version, we use invariant filters, norm max_pool, and no batch_norm.
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        key (jnp.random key): key for any layers requiring randomization
        train (bool): whether train mode or test mode, relevant for batch_norm
        batch_stats (dict): state for batch_norm, default None
        output_keys (tuple of tuples of ints): the output key types of the model. For the Shallow Water
            experiments, should be ((0,0),(1,0)) for the pressure/velocity form and ((0,0),(0,1)) for 
            the pressure/vorticity form.
        depth (int): the depth of the layers, defaults to 64
        activaton_f (function): the function that we pass to batch_scalar_activation
        equivariant (bool): whether to use the equivariant version of the model, defaults to False
        conv_filters (Layer): the conv filters used for the equivariant version
        upsample_filters (Layer): the conv filters used for the upsample layer of the equivariant version
        return_params (bool): whether we are mapping the params, or applying them
    """
    num_downsamples = 4
    num_conv = 2

    assert output_keys is not None

    batch_stats_idx = 0
    if equivariant:
        assert (conv_filters is not None) and (upsample_filters is not None)
        filter_info = { 'type': ml.CONV_FIXED, 'filters': conv_filters }
        filter_info_upsample = { 'type': ml.CONV_FIXED, 'filters': upsample_filters }
        target_keys = output_keys
    else:
        filter_info = { 'type': ml.CONV_FREE, 'M': 3 }
        filter_info_upsample = { 'type': ml.CONV_FREE, 'M': 2 }
        target_keys = ((0,0),)
        if batch_stats is None:
            batch_stats = defaultdict(lambda: { 'mean': None, 'var': None })

        # convert to channels of a scalar layer
        layer = layer.to_scalar_layer()

    # first we do the downsampling
    residual_layers = []
    for downsample in range(num_downsamples):
        layer, params, batch_stats, batch_stats_idx = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            { 'filter_info': filter_info, 'depth': depth*(2**downsample), 'target_keys': target_keys },
            batch_stats,
            batch_stats_idx,
            activation_f,
            mold_params=return_params,
        )
        residual_layers.append(layer)
        layer = ml.batch_max_pool(layer, 2, use_norm=equivariant)

    # bottleneck layer
    layer, params, batch_stats, batch_stats_idx = unet_conv_block(
        params, 
        layer, 
        train, 
        num_conv, 
        { 'filter_info': filter_info, 'depth': depth*(2**num_downsamples), 'target_keys': target_keys },
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
            filter_info_upsample,
            depth*(2**upsample),
            target_keys,
            padding=((1,1),)*layer.D,
            lhs_dilation=(2,)*layer.D, # do the transposed convolution
            mold_params=return_params,
        )
        # concat the upsampled layer and the residual
        layer = layer.concat(residual_layers[upsample], axis=1)

        layer, params, batch_stats, batch_stats_idx = unet_conv_block(
            params, 
            layer, 
            train, 
            num_conv, 
            { 'filter_info': filter_info, 'depth': depth*(2**upsample), 'target_keys': target_keys },
            batch_stats,
            batch_stats_idx,
            activation_f,
            mold_params=return_params,
        )

    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        filter_info,
        1, # depth
        output_keys,
        mold_params=return_params,
    )

    if (batch_stats is not None) or return_params:
        res = (layer,)
        if batch_stats is not None:
            res = res + (batch_stats,)
        if return_params:
            res = res + (params,)
    else:
        res = layer

    return res

@functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8,10,11])
def dil_resnet(
    params, 
    layer, 
    key, 
    train, 
    output_keys,
    output_depth=1,
    depth=48, 
    activation_f=jax.nn.relu, 
    equivariant=False, 
    conv_filters=None, 
    use_group_norm=False,
    return_params=False,
):
    """
    The DilatedResNet from https://arxiv.org/pdf/2112.15275.pdf. There are 4 dilated resnet blocks. There 
    7 dilated convoltions that are [1,2,4,8,4,2,1] with biases, each followed by an activation function.
    The residual layers are then added (not concatenated) back at the end of the block. There is a single 
    convolution (with bias) at the beginning and end to encode and decode. The 
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        key (jnp.random key): key for any layers requiring randomization
        train (bool): whether train mode or test mode, relevant for batch_norm
        output_keys (tuple of tuples of ints): the output key types of the model. For the Shallow Water
            experiments, should be ((0,0),(1,0)) for the pressure/velocity form and ((0,0),(0,1)) for 
            the pressure/vorticity form.
        output_depth (int or tuple): output depth for each output key. If an int, then the depth is 
            that int for each key. Defaults to 1, cannot be None. Needs to be hashable, so the structure
            is ( (key,depth), (key,depth)) where key itself is a tuple like (0,0)
        depth (int): the depth of the layers, defaults to 48
        activation_f (string or function): the function that we pass to batch_scalar_activation, defaults to relu
        equivariant (bool): whether to use the equivariant version of the model, defaults to False
        conv_filters (Layer): the conv filters used for the equivariant version
        return_params (bool): whether we are mapping the params, or applying them
    """
    assert layer.D == 2
    num_blocks = 4

    if isinstance(output_depth, int):
        output_depth = tuple((key,output_depth) for key in output_keys)

    if equivariant:
        assert conv_filters is not None
        filter_info_M3 = { 'type': ml.CONV_FIXED, 'filters': conv_filters }
        filter_info_M1 = filter_info_M3
        target_keys = output_keys
    else:
        layer = layer.to_scalar_layer()
        filter_info_M3 = { 'type': ml.CONV_FREE, 'M': 3 }
        filter_info_M1 = { 'type': ml.CONV_FREE, 'M': 1 }
        target_keys = ((0,0),)

    # encoder
    for _ in range(2):
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            filter_info_M1,
            depth,
            target_keys,
            bias=True,
            mold_params=return_params,
        )
        layer, params = handle_activation(activation_f, params, layer, return_params)

    for _ in range(num_blocks):

        residual_layer = layer.copy()
        # dCNN block
        for dilation in [1,2,4,8,4,2,1]:
            layer, params = ml.batch_conv_layer(
                params, 
                layer,
                filter_info_M3,
                depth,
                target_keys,
                bias=True,
                mold_params=return_params,
                rhs_dilation=(dilation,)*layer.D,
            )
            if use_group_norm:
                layer, params = ml.group_norm(params, layer, 1, equivariant=equivariant, mold_params=return_params)
            layer, params = handle_activation(activation_f, params, layer, return_params)

        layer = layer + residual_layer

    # decoder
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        filter_info_M1,
        depth,
        target_keys,
        bias=True,
        mold_params=return_params,
    )
    layer, params = handle_activation(activation_f, params, layer, return_params)
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        filter_info_M1,
        output_depth,
        output_keys,
        bias=True,
        mold_params=return_params,
    )

    return (layer, params) if return_params else layer

@functools.partial(jax.jit, static_argnums=[3,4,5,6,7,8,10,11])
def resnet(
    params, 
    layer, 
    key, 
    train, 
    output_keys,
    output_depth=1,
    depth=128, 
    activation_f=jax.nn.gelu,
    equivariant=False, 
    conv_filters=None, 
    use_group_norm=True,
    return_params=False,
):
    """
    The ResNet. There are 8 resnet blocks with 2 convolutions each, group norm, activations before each 
    convolution. 
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        key (jnp.random key): key for any layers requiring randomization
        train (bool): whether train mode or test mode, relevant for batch_norm
        output_keys (tuple of tuples of ints): the output key types of the model. For the Shallow Water
            experiments, should be ((0,0),(1,0)) for the pressure/velocity form and ((0,0),(0,1)) for 
            the pressure/vorticity form.
        output_depth (int or tuple): output depth for each output key. If an int, then the depth is 
            that int for each key. Defaults to 1, cannot be None. Needs to be hashable, so the structure
            is ( (key,depth), (key,depth)) where key itself is a tuple like (0,0)
        depth (int): the depth of the layers, defaults to 48
        activation_f (string or function): the function that we pass to batch_scalar_activation, defaults to relu
        equivariant (bool): whether to use the equivariant version of the model, defaults to False
        conv_filters (Layer): the conv filters used for the equivariant version
        return_params (bool): whether we are mapping the params, or applying them
    """
    num_blocks = 8
    num_conv = 2

    assert output_keys is not None
    assert output_depth is not None

    if isinstance(output_depth, int):
        output_depth = tuple((key,output_depth) for key in output_keys)

    if equivariant:
        assert conv_filters is not None
        filter_info_M3 = { 'type': ml.CONV_FIXED, 'filters': conv_filters }
        filter_info_M1 = filter_info_M3 # there aren't enough ones for this to work well
        target_keys = output_keys
    else:
        layer = layer.to_scalar_layer()
        filter_info_M3 = { 'type': ml.CONV_FREE, 'M': 3 }
        filter_info_M1 = { 'type': ml.CONV_FREE, 'M': 1 }
        target_keys = ((0,0),)

    # encoder
    for _ in range(2):
        layer, params = ml.batch_conv_layer(
            params, 
            layer,
            filter_info_M1,
            depth,
            target_keys,
            bias=True,
            mold_params=return_params,
        )
        layer, params = handle_activation(activation_f, params, layer, return_params)

    for _ in range(num_blocks):

        shortcut_layer = layer.copy()

        for _ in range(num_conv):
            # pre-activation order
            if use_group_norm:
                layer, params = ml.group_norm(params, layer, 1, equivariant=equivariant, mold_params=return_params)
            layer, params = handle_activation(activation_f, params, layer, return_params)

            layer, params = ml.batch_conv_layer(
                params, 
                layer,
                filter_info_M3,
                depth,
                target_keys,
                bias=True,
                mold_params=return_params,
            )

        layer = layer + shortcut_layer

    # decoder
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        filter_info_M1,
        depth,
        target_keys,
        bias=True,
        mold_params=return_params,
    )
    layer, params = handle_activation(activation_f, params, layer, return_params)
    
    layer, params = ml.batch_conv_layer(
        params, 
        layer,
        filter_info_M1,
        output_depth,
        output_keys,
        bias=True,
        mold_params=return_params,
    )

    return (layer, params) if return_params else layer

def do_nothing(params, layer, key, train, idxs, return_params=False):
    """
    Not to be confused with Calvin Coolidge. Allows you to compare the loss to a model that just picks one 
    channel of each layer as the output. This is helpful for dynamics problems where sometimes just picking
    the next step as the previous step is a surprisingly effective strategy.
    args:
        params (dict): the params tree
        layer (BatchLayer): input layer
        key (rand_key):
        train (bool): whether we are train mode
        idxs (dict): dict of (k,parity): idx where idx is layer to select as the output
    """
    out_layer = layer.empty()
    for (k, parity), image_block in layer.items():
        idx = idxs[(k,parity)]
        out_layer.append(k, parity, image_block[:,idx:idx+1])

    return (out_layer, params) if return_params else out_layer

def group_average(
    params, 
    layer, 
    key, 
    train, 
    model_f,
    *args,
    always_average=False,
    return_params=False,
    **kwargs,
): 
    if train and (not always_average):
        # train the model as normal
        return model_f(params, layer, key, train, *args, return_params=return_params, **kwargs)
    else:
        # during evaluation time, do the group averaging

        group_operators = geom.make_all_operators(layer.D)
        sum_layer = None
        for gg in group_operators:
            out = model_f(
                { k:v for k,v in params.items() }, 
                layer.times_group_element(gg), 
                key, 
                train, 
                *args,
                return_params=return_params,
                **kwargs,
            )

            if return_params:
                out, out_params = out

            out_rot = out.times_group_element(gg.T)
            sum_layer = out_rot if sum_layer is None else sum_layer + out_rot

        mean_layer = sum_layer / len(group_operators)
        return (mean_layer, out_params) if return_params else mean_layer

def unetBase_approx_equiv(
    params, 
    layer, 
    key, 
    train, 
    output_keys=None,
    depth=64, 
    activation_f=jax.nn.gelu,
    conv_filters=None, 
    upsample_filters=None, # 2x2 filters
    use_group_norm=True,
    scale_adjust=0.1,
    return_params=False,
):
    """
    The U-Net Base model that uses an equivariant model and a vanilla model. The involves 4 downsampling
    steps and 4 upsampling steps, 
    each interspersed with 2 convolutions interleaved with scalar activations and batch norm. The downsamples
    are achieved with max_pool layers with patch length of 2 and the upsamples are achieved with transposed
    convolution. Additionally, there are residual connections after every downsample and upsample. For the 
    equivariant version, we use invariant filters, norm max_pool, and no batch_norm.
    args:
        params (params tree): the params of the model
        layer (BatchLayer): input batch layer
        key (jnp.random key): key for any layers requiring randomization
        train (bool): whether train mode or test mode, relevant for batch_norm
        batch_stats (dict): state for batch_norm, default None
        output_keys (tuple of tuples of ints): the output key types of the model. For the Shallow Water
            experiments, should be ((0,0),(1,0)) for the pressure/velocity form and ((0,0),(0,1)) for 
            the pressure/vorticity form.
        depth (int): the depth of the layers, defaults to 64
        activation_f (string or function): the function that we pass to batch_scalar_activation
        equivariant (bool): whether to use the equivariant version of the model, defaults to False
        conv_filters (Layer): the conv filters used for the equivariant version
        upsample_filters (Layer): the conv filters used for the upsample layer of the equivariant version
        use_group_norm (bool): whether to use group_norm, defaults to True
        return_params (bool): whether we are mapping the params, or applying them
    """
    equiv_layer, params = unetBase(
        params, 
        layer, 
        key, 
        train, 
        output_keys,
        depth, 
        activation_f,
        True, # equivariant
        conv_filters, 
        upsample_filters, # 2x2 filters
        use_group_norm,
        return_params=True,
        mold_params=return_params,
    )

    free_result = unetBase(
        params, 
        layer, 
        key, 
        train, 
        output_keys,
        depth*2, 
        equivariant=False,
        return_params=return_params,
    )

    if return_params:
        free_layer = free_result[0]
        params = free_result[1]
    else:
        free_layer = free_result

    # scale the non-equivariant result to have 1/10th of max norm
    equiv_max_norm = jnp.max(equiv_layer.norm().to_vector())
    free_max_norm = jnp.max(free_layer.norm().to_vector())
    scale = (equiv_max_norm / free_max_norm)*scale_adjust

    if return_params:
        return equiv_layer + (free_layer*scale), params
    else:
        return equiv_layer + (free_layer*scale)
