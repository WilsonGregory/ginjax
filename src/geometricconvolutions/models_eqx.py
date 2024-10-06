from typing import Callable, Optional, Sequence, Union
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import equinox as eqx

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml_eqx as ml_eqx
import geometricconvolutions.ml as ml

ACTIVATION_REGISTRY = {
    "relu": jax.nn.relu,
    "gelu": jax.nn.gelu,
    "tanh": jax.nn.tanh,
}


def handle_activation(
    activation_f: Union[Callable, str],
    equivariant: bool,
    input_keys: tuple[tuple[ml.LayerKey, int]],
    D: int,
    key: ArrayLike,
):
    if equivariant:
        if activation_f is None:
            return lambda x: x
        elif isinstance(activation_f, str):
            return ml_eqx.VectorNeuronNonlinear(
                input_keys, D, ACTIVATION_REGISTRY[activation_f], key=key
            )
        else:
            return ml_eqx.VectorNeuronNonlinear(input_keys, D, activation_f, key=key)
    else:
        if activation_f is None:
            return ml_eqx.LayerWrapper(eqx.nn.Identity(), input_keys)
        elif isinstance(activation_f, str):
            return ml_eqx.LayerWrapper(ACTIVATION_REGISTRY[activation_f], input_keys)
        else:
            return ml_eqx.LayerWrapper(activation_f, input_keys)


def make_conv(
    D: int,
    input_keys: tuple[tuple[ml.LayerKey, int]],
    target_keys: tuple[tuple[ml.LayerKey, int]],
    use_bias: Union[str, bool],
    equivariant: bool,
    invariant_filters: Optional[geom.Layer] = None,
    kernel_size: Optional[Union[int, Sequence[int]]] = None,
    stride: Optional[tuple[int]] = None,
    padding: Optional[tuple[int]] = None,
    lhs_dilation: Optional[Union[tuple[int], bool]] = None,
    rhs_dilation: Optional[tuple[int]] = None,
    key: Optional[ArrayLike] = None,
):
    """
    Factory for convolution layer which makes ConvContract if equivariant and makes a regular conv
    otherwise.
    """
    if equivariant:
        return ml_eqx.ConvContract(
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
        # TODO: need to add a reshaper layer?
        assert len(input_keys) == len(target_keys) == 1
        assert input_keys[0][0] == target_keys[0][0] == (0, 0)
        padding = "SAME" if padding is None else padding
        stride = 1 if stride is None else stride
        rhs_dilation = 1 if rhs_dilation is None else rhs_dilation
        if lhs_dilation is None:
            return ml_eqx.LayerWrapper(
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
            return ml_eqx.LayerWrapper(
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
    return sum(
        [
            0 if x is None else x.size
            for x in eqx.filter(jax.tree_util.tree_leaves(model), eqx.is_array)
        ]
    )


class UNet(eqx.Module):
    embedding: list[eqx.Module]
    downsample_blocks: list[list[eqx.Module]]
    upsample_blocks: list[list[eqx.Module]]
    decode: ml_eqx.ConvContract

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    output_keys: tuple[tuple[ml.LayerKey, int]] = eqx.field(static=True)

    def __init__(
        self: Self,
        D,
        input_keys: tuple[tuple[ml.LayerKey, int]],
        output_keys: tuple[tuple[ml.LayerKey, int]],
        depth: int,
        num_downsamples: int = 4,
        num_conv: int = 2,
        use_bias: Union[bool, str] = "auto",
        activation_f: Union[Callable, str] = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: geom.Layer = None,
        upsample_filters: geom.Layer = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        key: ArrayLike = None,
    ):
        assert num_conv > 0

        self.output_keys = output_keys
        if equivariant:
            key_union = {k_p for k_p, _ in input_keys}.union({k_p for k_p, _ in output_keys})
            mid_keys = tuple((k_p, depth) for k_p in key_union)
        else:
            mid_keys = (((0, 0), depth),)
            # use these keys along the way, then for the final output use self.output_keys
            input_keys = (((0, 0), sum(in_c * (D**k) for (k, _), in_c in input_keys)),)
            output_keys = (((0, 0), sum(out_c * (D**k) for (k, _), out_c in output_keys)),)

        self.D = D
        self.equivariant = equivariant

        # embedding layers
        self.embedding = []
        for conv_idx in range(num_conv):
            in_keys = input_keys if conv_idx == 0 else mid_keys
            key, subkey1, subkey2 = random.split(key, num=3)
            self.embedding.append(
                make_conv(
                    self.D,
                    in_keys,
                    mid_keys,
                    use_bias,
                    equivariant,
                    conv_filters,
                    kernel_size,
                    key=subkey1,
                )
            )

            if use_group_norm:
                self.embedding.append(ml_eqx.LayerNorm(mid_keys, self.D))

            self.embedding.append(
                handle_activation(activation_f, self.equivariant, mid_keys, self.D, subkey2)
            )

        self.downsample_blocks = []
        for downsample in range(1, num_downsamples + 1):
            down_layers = [ml_eqx.MaxNormPool(2, equivariant)]

            for conv_idx in range(num_conv):
                out_keys = tuple((k_p, depth * (2**downsample)) for k_p, _ in mid_keys)
                if conv_idx == 0:
                    in_keys = tuple((k_p, depth * (2 ** (downsample - 1))) for k_p, _ in mid_keys)
                else:
                    in_keys = out_keys

                key, subkey1, subkey2 = random.split(key, num=3)
                down_layers.append(
                    make_conv(
                        self.D,
                        in_keys,
                        out_keys,
                        use_bias,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        key=subkey1,
                    )
                )
                if use_group_norm:
                    down_layers.append(ml_eqx.LayerNorm(out_keys, self.D))

                down_layers.append(
                    handle_activation(activation_f, self.equivariant, out_keys, self.D, subkey2)
                )

            self.downsample_blocks.append(down_layers)

        self.upsample_blocks = []
        for upsample in reversed(range(num_downsamples)):
            in_keys = tuple((k_p, depth * (2 ** (upsample + 1))) for k_p, _ in mid_keys)
            out_keys = tuple((k_p, depth * (2**upsample)) for k_p, _ in mid_keys)
            key, subkey = random.split(key)
            # perform the transposed convolution. For non-equivariant, padding and stride should
            # instead be the padding and stride for the forward direction convolution.
            if equivariant:
                padding = ((1, 1),) * self.D
                lhs_dilation = (2,) * self.D
                stride = None
                upsample_kernel_size = None  # ignored for equivariant
            else:
                padding = "VALID"
                lhs_dilation = True  # signals to do ConvTranspose
                stride = (2,) * self.D
                upsample_kernel_size = (2,) * self.D  # kernel size of the downsample

            up_layers = [
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
                    lhs_dilation,
                    key=subkey,
                )
            ]

            for conv_idx in range(num_conv):
                out_keys = tuple((k_p, depth * (2**upsample)) for k_p, _ in mid_keys)
                if conv_idx == 0:  # due to adding the residual layer back, in_c is doubled again
                    in_keys = tuple((k_p, depth * (2 ** (upsample + 1))) for k_p, _ in mid_keys)
                else:
                    in_keys = out_keys

                key, subkey1, subkey2 = random.split(key, num=3)
                up_layers.append(
                    make_conv(
                        self.D,
                        in_keys,
                        out_keys,
                        use_bias,
                        equivariant,
                        conv_filters,
                        kernel_size,
                        key=subkey1,
                    )
                )
                if use_group_norm:
                    up_layers.append(ml_eqx.LayerNorm(out_keys, self.D))

                up_layers.append(
                    handle_activation(activation_f, self.equivariant, out_keys, self.D, subkey2)
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

    def __call__(self: Self, x: geom.Layer):
        if not self.equivariant:
            in_layer = x
            # to_scalar_layer is not working well, so do this
            spatial_dims, _ = geom.parse_shape(x[(0, 0)].shape[1:], x.D)
            data_arr = jnp.zeros((0,) + spatial_dims)
            for (k, _), image in in_layer.items():
                transpose_idxs = (
                    (0,) + tuple(range(1 + self.D, 1 + self.D + k)) + tuple(range(1, 1 + self.D))
                )
                data_arr = jnp.concatenate(
                    [data_arr, image.transpose(transpose_idxs).reshape((-1,) + spatial_dims)]
                )
            x = geom.BatchLayer({(0, 0): data_arr}, in_layer.D, in_layer.is_torus)

        for layer in self.embedding:
            x = layer(x)

        residual_layers = []
        for block in self.downsample_blocks:
            residual_layers.append(x)
            for layer in block:
                x = layer(x)

        for block, residual_idx in zip(self.upsample_blocks, reversed(range(len(residual_layers)))):
            upsample_x = block[0](x)  # first layer in block is the upsample
            x = upsample_x.concat(residual_layers[residual_idx])
            for layer in block[1:]:
                x = layer(x)

        x = self.decode(x)
        if self.equivariant:
            return x
        else:
            output_keys = {(k, p): out_c for (k, p), out_c in self.output_keys}
            out_layer = geom.Layer.from_scalar_layer(x, output_keys)
            # convert back to a vmapped batch layer
            return geom.BatchLayer(out_layer.data, out_layer.D, out_layer.is_torus)
