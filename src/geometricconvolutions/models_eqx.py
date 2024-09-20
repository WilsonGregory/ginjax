from typing import Callable, Optional, Sequence, Union
from typing_extensions import Self

import jax
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
    input_keys: tuple[tuple[ml.LayerKey, int]],
    D: int,
    key: ArrayLike,
):
    if activation_f is None:
        return eqx.nn.Identity()
    elif activation_f == ml.VN_NONLINEAR:
        return ml_eqx.VectorNeuronNonlinear(input_keys, D, key=key)
    elif isinstance(activation_f, str):
        return lambda x: ml.batch_scalar_activation(x, ACTIVATION_REGISTRY[activation_f])
    else:
        return lambda x: ml.batch_scalar_activation(x, activation_f)


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
    lhs_dilation: Optional[tuple[int]] = None,
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
        in_c = sum(in_c for _, in_c in input_keys)
        out_c = sum(out_c for _, out_c in target_keys)
        padding = "SAME" if padding is None else padding
        if lhs_dilation is None:
            return eqx.nn.Conv(
                D,
                in_c,
                out_c,
                kernel_size,
                stride,
                padding,
                rhs_dilation,
                use_bias=use_bias,
                key=key,
            )
        else:  # currently ignores actual value of lhs_dilation, hopefully this works
            return eqx.nn.ConvTranspose(
                D,
                in_c,
                out_c,
                kernel_size,
                stride,
                padding,
                dilation=rhs_dilation,
                use_bias=use_bias,
                key=key,
            )


def count_params(model: eqx.Module) -> int:
    return sum([x.size for x in jax.tree_util.tree_leaves(model)])


class UNet(eqx.Module):
    embedding: list[eqx.Module]
    downsample_blocks: list[list[eqx.Module]]
    upsample_blocks: list[list[eqx.Module]]
    decode: ml_eqx.ConvContract

    D: int = eqx.field(static=True)

    def __init__(
        self: Self,
        D,
        input_keys: tuple[tuple[ml.LayerKey, int]],
        output_keys: tuple[tuple[ml.LayerKey, int]],
        depth: int,
        num_downsamples: int = 4,
        num_conv: int = 2,
        use_bias: Union[bool, str] = "auto",
        activation_f: Union[Callable, str] = ml.VN_NONLINEAR,
        equivariant: bool = True,
        conv_filters: geom.Layer = None,
        upsample_filters: geom.Layer = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        key: ArrayLike = None,
    ):
        assert num_conv > 0
        if equivariant:
            key_union = {k_p for k_p, _ in input_keys}.union({k_p for k_p, _ in output_keys})
            mid_keys = tuple((k_p, depth) for k_p in key_union)
        else:
            mid_keys = (((0, 0), depth),)

        self.D = D

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

            self.embedding.append(handle_activation(activation_f, mid_keys, self.D, subkey2))

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

                down_layers.append(handle_activation(activation_f, out_keys, self.D, subkey2))

            self.downsample_blocks.append(down_layers)

        self.upsample_blocks = []
        for upsample in reversed(range(num_downsamples)):
            in_keys = tuple((k_p, depth * (2 ** (upsample + 1))) for k_p, _ in mid_keys)
            out_keys = tuple((k_p, depth * (2**upsample)) for k_p, _ in mid_keys)
            key, subkey = random.split(key)
            # perform the transposed convolution
            up_layers = [
                make_conv(
                    self.D,
                    in_keys,
                    out_keys,
                    use_bias,
                    equivariant,
                    upsample_filters,
                    kernel_size,
                    padding=((1, 1),) * self.D,
                    lhs_dilation=(2,) * self.D,
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
                        padding=((1, 1),) * self.D,
                        lhs_dilation=(2,) * self.D,
                        key=subkey1,
                    )
                )
                if use_group_norm:
                    up_layers.append(ml_eqx.LayerNorm(out_keys, self.D))

                up_layers.append(handle_activation(activation_f, out_keys, self.D, subkey2))

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

        return self.decode(x)
