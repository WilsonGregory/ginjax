from typing import Any, Callable, Optional, Sequence, Union
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml

ACTIVATION_REGISTRY = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
}


def handle_activation(
    activation_f: Optional[Union[Callable, str]],
    equivariant: bool,
    input_keys: geom.Signature,
    D: int,
    key: ArrayLike,
) -> Callable[[Any], geom.MultiImage]:
    """
    Parse what activation function to use, return the appropriate callable

    args:
        activation_f: the type of activation, either a callable or a string name from ACTIVATION_REGISTRY
        equivariant: whether to use an equivariant activation
        input_keys: the layers input keys
        D: dimension of the model
        key: jax.random key

    returns:
        A layer that performs the specified activation function
    """
    if equivariant:
        if activation_f is None:
            return lambda x: x
        elif isinstance(activation_f, str):
            assert activation_f in ACTIVATION_REGISTRY
            return ml.VectorNeuronNonlinear(
                input_keys, D, ACTIVATION_REGISTRY[activation_f], key=key
            )
        else:
            return ml.VectorNeuronNonlinear(input_keys, D, activation_f, key=key)
    else:
        if activation_f is None:
            return ml.LayerWrapper(eqx.nn.Identity(), input_keys)
        elif isinstance(activation_f, str):
            assert activation_f in ACTIVATION_REGISTRY
            return ml.LayerWrapper(ACTIVATION_REGISTRY[activation_f], input_keys)
        else:
            return ml.LayerWrapper(activation_f, input_keys)


def make_conv(
    D: int,
    input_keys: geom.Signature,
    target_keys: geom.Signature,
    use_bias: Union[str, bool],
    equivariant: bool,
    invariant_filters: Optional[geom.MultiImage] = None,
    kernel_size: Optional[Union[int, Sequence[int]]] = None,
    stride: Union[tuple[int, ...], int] = 1,
    padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = None,
    lhs_dilation: Optional[tuple[int, ...]] = None,
    rhs_dilation: Union[int, tuple[int, ...]] = 1,
    key: Any = None,  # any instead of arraylike because split cannot handle None
) -> Union[ml.ConvContract, ml.LayerWrapper]:
    """
    Factory for convolution layer which makes ConvContract if equivariant and makes a regular conv
    otherwise.

    args:
        D: dimension of the space
        input_keys: MultiImage Signature of input
        target_keys: MultiImage Signature of output
        use_bias: whether to use a bias
        equivariant: whether to use an equivariant layer or normal layer
        invariant_filters: filters used for equivariant layer
        kernel_size: sidelength(s) of kernel, only used for non-equivariant layer
        stride: convolution stride
        padding: convolution padding
        lhs_dilation: left hand side dilation for transpose convolution
        rhs_dilation: right hand side dilation for dilated convolutions
        key: jax.random key

    returns:
        either ConvContract or a LayerWrapper around an equinox convolution
    """
    if equivariant:
        assert invariant_filters is not None
        return ml.ConvContract(
            input_keys,
            target_keys,
            invariant_filters,
            use_bias,
            stride,
            padding,
            lhs_dilation,
            rhs_dilation,
            key,
        )
    else:
        assert kernel_size is not None
        assert len(input_keys) == len(target_keys) == 1
        assert input_keys[0][0] == target_keys[0][0] == (0, 0)
        padding = "SAME" if padding is None else padding
        use_bias = True if use_bias == "auto" else use_bias
        assert isinstance(use_bias, bool)
        if lhs_dilation is None:
            return ml.LayerWrapper(
                eqx.nn.Conv(
                    D,
                    input_keys[0][1],
                    target_keys[0][1],
                    kernel_size,
                    stride,
                    padding,
                    rhs_dilation,
                    use_bias=use_bias,
                    key=key,
                ),
                input_keys,
            )
        else:
            # if there is lhs_dilation, assume its a transpose convolution
            return ml.LayerWrapper(
                eqx.nn.ConvTranspose(
                    D,
                    input_keys[0][1],
                    target_keys[0][1],
                    kernel_size,
                    stride,
                    padding,
                    dilation=rhs_dilation,
                    use_bias=use_bias,
                    key=key,
                ),
                input_keys,
            )


def count_params(model: eqx.Module) -> int:
    """
    Count the number of parameters in the model

    args:
        model: model to measure

    returns:
        number of parameters
    """
    return sum(
        [
            0 if x is None else x.size
            for x in eqx.filter(jax.tree_util.tree_leaves(model), eqx.is_array)
        ]
    )


class ConvBlock(eqx.Module):
    """
    A convolution block consisting of a convolution, a nonlinearity, and a GroupNorm/BatchNorm.
    Can be equivariant or not, in typical order or in preactivation order.
    """

    conv: Union[ml.ConvContract, ml.LayerWrapper]
    group_norm: Optional[Union[ml.GroupNorm, ml.LayerWrapper]]
    batch_norm: Optional[ml.LayerWrapperAux]
    nonlinearity: Union[ml.VectorNeuronNonlinear, ml.LayerWrapper, Callable]

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    use_batch_norm: bool = eqx.field(static=True)
    use_group_norm: bool = eqx.field(static=True)
    preactivation_order: bool = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        use_bias: Union[bool, str] = "auto",
        activation_f: Optional[Union[Callable, str]] = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: Optional[geom.MultiImage] = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        use_batch_norm: bool = False,
        preactivation_order: bool = False,
        key: Any = None,
        **conv_kwargs: Any,
    ) -> None:
        """
        Constructor for ConvBlock

        args:
            D: the dimension of the space
            input_keys: MultiImage Signature of input
            output_keys: MultiImage Signature of output
            use_bias: whether to use a bias
            activation_f: the type of activation function
            equivariant: whether it is equivariant
            conv_filters: the invariant filters if it is equivariant
            kernel_size: sidelength(s) of the kernel if not equivariant
            use_group_norm: whether to use GroupNorm
            use_batch_norm: whether to use BatchNorm, can only be for non-equivariant
            preactivation_order: whether to use preactivation order
            key: jax.random key
            conv_kwargs: further key word args that will be passed to the convolution
        """
        self.D = D
        self.equivariant = equivariant
        self.use_group_norm = use_group_norm
        self.use_batch_norm = use_batch_norm
        self.preactivation_order = preactivation_order

        subkey1, subkey2 = random.split(key)
        self.conv = make_conv(
            self.D,
            input_keys,
            output_keys,
            use_bias,
            equivariant,
            conv_filters,
            kernel_size,
            key=subkey1,
            **conv_kwargs,
        )

        if use_group_norm:
            if self.equivariant:
                self.group_norm = ml.LayerNorm(output_keys, self.D)
            else:
                self.group_norm = ml.LayerWrapper(
                    eqx.nn.GroupNorm(1, output_keys[0][1]), output_keys
                )
        else:
            self.group_norm = None

        if use_batch_norm:
            self.batch_norm = ml.LayerWrapperAux(
                eqx.nn.BatchNorm(output_keys[0][1], axis_name=["pmap_batch", "batch"]), output_keys
            )
        else:
            self.batch_norm = None

        self.nonlinearity = handle_activation(
            activation_f, self.equivariant, output_keys, self.D, subkey2
        )

    def __call__(
        self: Self, x: geom.MultiImage, batch_stats: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        """
        Layer callable

        args:
            x: the input
            batch_stats: data for batch norm

        returns:
            the output MultiImage and batch stats
        """
        if self.preactivation_order:
            if self.use_group_norm:
                assert self.group_norm is not None
                x = self.group_norm(x)
            elif self.use_batch_norm:
                assert self.batch_norm is not None
                x, batch_stats = self.batch_norm(x, batch_stats)

            x = self.nonlinearity(x)
            x = self.conv(x)
        else:
            x = self.conv(x)
            if self.use_group_norm:
                assert self.group_norm is not None
                x = self.group_norm(x)
            elif self.use_batch_norm:
                assert self.batch_norm is not None
                x, batch_stats = self.batch_norm(x, batch_stats)

            x = self.nonlinearity(x)

        return x, batch_stats


class UNet(eqx.Module):
    """
    Implementation of the UNet: https://arxiv.org/abs/1505.04597.
    This model defaults to the equivariant version, but can also be the non-equivariant version.
    """

    embedding: list[ConvBlock]
    downsample_blocks: list[tuple[ml.MaxNormPool, list[ConvBlock]]]
    upsample_blocks: list[tuple[Union[ml.ConvContract, ml.LayerWrapper], list[ConvBlock]]]
    decode: Union[ml.ConvContract, ml.LayerWrapper]

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    use_batch_norm: bool = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_downsamples: int = 4,
        num_conv: int = 2,
        use_bias: Union[bool, str] = "auto",
        activation_f: Union[Callable, str] = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: Optional[geom.MultiImage] = None,
        upsample_filters: Optional[geom.MultiImage] = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        use_batch_norm: bool = False,
        key: Any = None,
    ) -> None:
        """
        Constructor for the UNet.

        args:
            D: the dimension of the space
            input_keys: the MultiImage Signature for the input
            output_keys: the MultiImage Signature for the output
            depth: the number of channelsat the highest level of the unet
            num_downsamples: number of convolution blocks followed by a max pool
            num_conv: number of convolutions per level
            use_bias: whether to use a bias
            activation_f: the activation function
            equivariant: whether to be equivariant
            conv_filters: the invariant filters for the equivariant version
            kernel_size: sidelength(s) for the non-equivariant version
            use_group_norm: whether to use GroupNorm
            use_batch_norm: whether to use the BatchNorm, only for non-equivariant version
            key: jax.random key
        """
        assert num_conv > 0
        assert key is not None

        self.output_keys = output_keys
        if equivariant:
            key_union = {k_p for k_p, _ in input_keys}.union({k_p for k_p, _ in output_keys})
            mid_keys = geom.Signature(tuple((k_p, depth) for k_p in key_union))
            assert not use_batch_norm, "UNet::init Batch Norm cannot be used with equivariant model"
        else:
            mid_keys = geom.Signature((((0, 0), depth),))
            # use these keys along the way, then for the final output use self.output_keys
            input_keys_size = sum(in_c * (D**k) for (k, _), in_c in input_keys)
            input_keys = geom.Signature((((0, 0), input_keys_size),))
            output_key_size = sum(out_c * (D**k) for (k, _), out_c in output_keys)
            output_keys = geom.Signature((((0, 0), output_key_size),))

        self.D = D
        self.equivariant = equivariant
        self.use_batch_norm = use_batch_norm

        # embedding layers
        self.embedding = []
        for conv_idx in range(num_conv):
            in_keys = input_keys if conv_idx == 0 else mid_keys
            key, subkey = random.split(key)
            self.embedding.append(
                ConvBlock(
                    self.D,
                    in_keys,
                    mid_keys,
                    use_bias,
                    activation_f,
                    equivariant,
                    conv_filters,
                    kernel_size,
                    use_group_norm,
                    use_batch_norm,
                    key=subkey,
                )
            )

        self.downsample_blocks = []
        for downsample in range(1, num_downsamples + 1):
            down_layers = (ml.MaxNormPool(2, equivariant), [])

            for conv_idx in range(num_conv):
                out_keys = geom.Signature(
                    tuple((k_p, depth * (2**downsample)) for k_p, _ in mid_keys)
                )
                if conv_idx == 0:
                    in_keys = geom.Signature(
                        tuple((k_p, depth * (2 ** (downsample - 1))) for k_p, _ in mid_keys)
                    )
                else:
                    in_keys = out_keys

                key, subkey = random.split(key)
                down_layers[1].append(
                    ConvBlock(
                        self.D,
                        in_keys,
                        out_keys,
                        use_bias,
                        activation_f,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        use_group_norm,
                        use_batch_norm,
                        key=subkey,
                    )
                )

            self.downsample_blocks.append(down_layers)

        self.upsample_blocks = []
        for upsample in reversed(range(num_downsamples)):
            in_keys = geom.Signature(
                tuple((k_p, depth * (2 ** (upsample + 1))) for k_p, _ in mid_keys)
            )
            out_keys = geom.Signature(tuple((k_p, depth * (2**upsample)) for k_p, _ in mid_keys))
            key, subkey = random.split(key)
            # perform the transposed convolution. For non-equivariant, padding and stride should
            # instead be the padding and stride for the forward direction convolution.
            if equivariant:
                padding = ((1, 1),) * self.D
                stride = (1,) * self.D
                upsample_kernel_size = None  # ignored for equivariant
            else:
                padding = "VALID"
                stride = (2,) * self.D
                upsample_kernel_size = (2,) * self.D  # kernel size of the downsample

            up_layers = (
                make_conv(
                    self.D,
                    in_keys,
                    out_keys,
                    use_bias,
                    equivariant,
                    upsample_filters,
                    upsample_kernel_size,
                    stride,
                    padding,
                    (2,) * self.D,  # lhs_dilation
                    key=subkey,
                ),
                [],
            )

            for conv_idx in range(num_conv):
                out_keys = geom.Signature(
                    tuple((k_p, depth * (2**upsample)) for k_p, _ in mid_keys)
                )
                if conv_idx == 0:  # due to adding the residual layer back, in_c is doubled again
                    in_keys = geom.Signature(
                        tuple((k_p, depth * (2 ** (upsample + 1))) for k_p, _ in mid_keys)
                    )
                else:
                    in_keys = out_keys

                key, subkey = random.split(key)
                up_layers[1].append(
                    ConvBlock(
                        self.D,
                        in_keys,
                        out_keys,
                        use_bias,
                        activation_f,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        use_group_norm,
                        use_batch_norm,
                        key=subkey,
                    )
                )

            self.upsample_blocks.append(up_layers)

        key, subkey = random.split(key)

        self.decode = make_conv(
            self.D,
            mid_keys,
            output_keys,
            use_bias,
            equivariant,
            conv_filters,
            kernel_size,
            key=subkey,
        )

    def __call__(
        self: Self, x: geom.MultiImage, batch_stats: Optional[eqx.nn.State] = None
    ) -> Union[geom.MultiImage, tuple[geom.MultiImage, Optional[eqx.nn.State]]]:
        """
        Callable function for UNet

        args:
            x: the input MultiImage
            batch_stats: batch stats for BatchNorm if present

        returns:
            the output MultiImage and batch_stats
        """
        if not self.equivariant:
            x = x.to_scalar_multi_image()

        for layer in self.embedding:
            x, batch_stats = layer(x, batch_stats)

        residual_multi_images = []
        for max_pool_layer, conv_blocks in self.downsample_blocks:
            residual_multi_images.append(x)
            x = max_pool_layer(x)
            for layer in conv_blocks:
                x, batch_stats = layer(x, batch_stats)

        for (upsample_layer, conv_blocks), residual_multi_image in zip(
            self.upsample_blocks, reversed(residual_multi_images)
        ):
            upsample_x = upsample_layer(x)
            x = upsample_x.concat(residual_multi_image)
            for layer in conv_blocks:
                x, batch_stats = layer(x, batch_stats)

        x = self.decode(x)
        if self.equivariant:
            out = x
        else:
            out = geom.MultiImage.from_scalar_multi_image(x, self.output_keys)

        return (out, batch_stats) if self.use_batch_norm else out


class DilResNet(eqx.Module):
    """
    The Dilated ResNet from https://arxiv.org/abs/2112.15275.
    """

    encoder: list[ConvBlock]
    blocks: list[list[ConvBlock]]
    decoder: list[ConvBlock]

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_blocks: int = 4,
        use_bias: Union[bool, str] = "auto",
        activation_f: Optional[Union[Callable, str]] = jax.nn.relu,
        equivariant: bool = True,
        conv_filters: Optional[geom.MultiImage] = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        key: Any = None,
    ) -> None:
        """
        Constructor for the DilatedResNet

        args:
            D: the dimension of the space
            input_keys: the MultiImage Signature for the input
            output_keys: the MultiImage Signature for the output
            depth: the number of channelsat the highest level of the unet
            num_blocks: number of resnet blocks
            use_bias: whether to use a bias
            activation_f: the activation function
            equivariant: whether to be equivariant
            conv_filters: the invariant filters for the equivariant version
            kernel_size: sidelength(s) for the non-equivariant version
            use_group_norm: whether to use GroupNorm
            key: jax.random key
        """
        self.D = D
        self.equivariant = equivariant
        self.output_keys = output_keys

        if equivariant:
            key_union = {k_p for k_p, _ in input_keys}.union({k_p for k_p, _ in output_keys})
            mid_keys = geom.Signature(tuple((k_p, depth) for k_p in key_union))
        else:
            mid_keys = geom.Signature((((0, 0), depth),))
            # use these keys along the way, then for the final output use self.output_keys
            input_keys = geom.Signature(
                (((0, 0), sum(in_c * (D**k) for (k, _), in_c in input_keys)),)
            )
            output_keys = geom.Signature(
                (((0, 0), sum(out_c * (D**k) for (k, _), out_c in output_keys)),)
            )

        # encoder
        key, subkey1, subkey2 = random.split(key, num=3)
        self.encoder = [
            ConvBlock(
                D,
                input_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey1,
            ),
            ConvBlock(
                D,
                mid_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey2,
            ),
        ]

        self.blocks = []
        for _ in range(num_blocks):
            # dCNN block
            dilation_block = []
            for dilation in [1, 2, 4, 8, 4, 2, 1]:
                key, subkey = random.split(key)
                dilation_block.append(
                    ConvBlock(
                        D,
                        mid_keys,
                        mid_keys,
                        use_bias,
                        activation_f,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        use_group_norm,
                        rhs_dilation=(dilation,) * D,
                        key=subkey,
                    )
                )

            self.blocks.append(dilation_block)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.decoder = [
            ConvBlock(
                D,
                mid_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey1,
            ),
            ConvBlock(
                D, mid_keys, output_keys, use_bias, None, equivariant, conv_filters, 1, key=subkey2
            ),
        ]

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for this layer

        args:
            x: the input MultiImage

        returns:
            the output MultiImage
        """
        if not self.equivariant:
            x = x.to_scalar_multi_image()

        for layer in self.encoder:
            x, _ = layer(x)

        for dilation_block in self.blocks:
            residual_x = x.copy()

            for layer in dilation_block:
                x, _ = layer(x)

            x = x + residual_x

        for layer in self.decoder:
            x, _ = layer(x)

        if self.equivariant:
            out = x
        else:
            out = geom.MultiImage.from_scalar_multi_image(x, self.output_keys)

        return out


class ResNet(eqx.Module):
    """
    A typical ResNet.
    """

    encoder: list[ConvBlock]
    blocks: list[list[ConvBlock]]
    decoder: list[ConvBlock]

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_blocks: int = 8,
        num_conv: int = 2,
        use_bias: Union[bool, str] = "auto",
        activation_f: Union[Callable, str] = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: Optional[geom.MultiImage] = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = True,
        preactivation_order: bool = True,
        key: Any = None,
    ) -> None:
        """
        Constructor for the ResNet

        args:
            D: the dimension of the space
            input_keys: the MultiImage Signature for the input
            output_keys: the MultiImage Signature for the output
            depth: the number of channelsat the highest level of the unet
            num_blocks: number of resnet blocks
            num_conv: number of convolutions per block
            use_bias: whether to use a bias
            activation_f: the activation function
            equivariant: whether to be equivariant
            conv_filters: the invariant filters for the equivariant version
            kernel_size: sidelength(s) for the non-equivariant version
            use_group_norm: whether to use GroupNorm
            preactivation_order: whether to use preactivation order
            key: jax.random key
        """
        self.D = D
        self.equivariant = equivariant
        self.output_keys = output_keys

        if equivariant:
            key_union = {k_p for k_p, _ in input_keys}.union({k_p for k_p, _ in output_keys})
            mid_keys = geom.Signature(tuple((k_p, depth) for k_p in key_union))
        else:
            mid_keys = geom.Signature((((0, 0), depth),))
            # use these keys along the way, then for the final output use self.output_keys
            input_keys = geom.Signature(
                (((0, 0), sum(in_c * (D**k) for (k, _), in_c in input_keys)),)
            )
            output_keys = geom.Signature(
                (((0, 0), sum(out_c * (D**k) for (k, _), out_c in output_keys)),)
            )

        # encoder
        key, subkey1, subkey2 = random.split(key, num=3)
        self.encoder = [
            ConvBlock(
                D,
                input_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey1,
            ),
            ConvBlock(
                D,
                mid_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey2,
            ),
        ]

        self.blocks = []
        for _ in range(num_blocks):
            # dCNN block
            block = []
            for _ in range(num_conv):
                key, subkey = random.split(key)
                block.append(
                    ConvBlock(
                        D,
                        mid_keys,
                        mid_keys,
                        use_bias,
                        activation_f,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        use_group_norm,
                        preactivation_order=preactivation_order,
                        key=subkey,
                    )
                )

            self.blocks.append(block)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.decoder = [
            ConvBlock(
                D,
                mid_keys,
                mid_keys,
                use_bias,
                activation_f,
                equivariant,
                conv_filters,
                1,
                key=subkey1,
            ),
            ConvBlock(
                D, mid_keys, output_keys, use_bias, None, equivariant, conv_filters, 1, key=subkey2
            ),
        ]

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for this layer

        args:
            x: the input MultiImage

        returns:
            the output MultiImage
        """
        if not self.equivariant:
            x = x.to_scalar_multi_image()

        for layer in self.encoder:
            x, _ = layer(x)

        for block in self.blocks:
            residual_x = x.copy()

            for layer in block:
                x, _ = layer(x)

            x = x + residual_x

        for layer in self.decoder:
            x, _ = layer(x)

        if self.equivariant:
            out = x
        else:
            out = geom.MultiImage.from_scalar_multi_image(x, self.output_keys)

        return out
