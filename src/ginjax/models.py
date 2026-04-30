import math
import numpy as np
from typing import Any, Callable, Optional, Sequence, Union
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import equinox as eqx

import ginjax.geometric as geom
from ginjax import layers

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
            return layers.VectorNeuronNonlinear(
                input_keys, D, ACTIVATION_REGISTRY[activation_f], key=key
            )
        else:
            return layers.VectorNeuronNonlinear(input_keys, D, activation_f, key=key)
    else:
        if activation_f is None:
            return layers.LayerWrapper(eqx.nn.Identity(), input_keys)
        elif isinstance(activation_f, str):
            assert activation_f in ACTIVATION_REGISTRY
            return layers.LayerWrapper(ACTIVATION_REGISTRY[activation_f], input_keys)
        else:
            return layers.LayerWrapper(activation_f, input_keys)


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
    padding_mode: str = "ZEROS",
    key: Any = None,  # any instead of arraylike because split cannot handle None
) -> Union[layers.ConvContract, layers.LayerWrapper]:
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
        padding_mode: for non-equivariant convolutions, define padding mode that is passed to conv.
            For equivariant, this is a variable of the input
        key: jax.random key

    returns:
        either ConvContract or a LayerWrapper around an equinox convolution
    """
    if equivariant:
        assert invariant_filters is not None
        return layers.ConvContract(
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
        assert input_keys[0][0] == target_keys[0][0] == ((), 0)
        padding = "SAME" if padding is None else padding
        padding_mode = padding_mode if padding == "SAME" else "ZEROS"  # only implemented for SAME
        use_bias = True if use_bias == "auto" else use_bias
        assert isinstance(use_bias, bool)
        if lhs_dilation is None:
            return layers.LayerWrapper(
                eqx.nn.Conv(
                    D,
                    input_keys[0][1],
                    target_keys[0][1],
                    kernel_size,
                    stride,
                    padding,
                    rhs_dilation,
                    use_bias=use_bias,
                    padding_mode=padding_mode,
                    key=key,
                ),
                input_keys,
            )
        else:
            # if there is lhs_dilation, assume its a transpose convolution
            return layers.LayerWrapper(
                eqx.nn.ConvTranspose(
                    D,
                    input_keys[0][1],
                    target_keys[0][1],
                    kernel_size,
                    stride,
                    padding,
                    dilation=rhs_dilation,
                    use_bias=use_bias,
                    padding_mode=padding_mode,
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
    # get the filters
    is_conv = lambda n: isinstance(n, layers.ConvContract)
    get_filters = lambda m: [
        x.invariant_filters for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
    ]
    get_array_sizes = lambda m: [
        x.size for x in jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array) if eqx.is_array(x)
    ]

    # filters are arrays, but they aren't params so we subtract them from the total array size
    total_size = sum(get_array_sizes(model))
    filter_size = sum(get_array_sizes(get_filters(model)))
    return total_size - filter_size


class MultiImageModule(eqx.Module):
    """
    A model that takes as input and output a MultiImage and aux_data. The models that inherit from
    this class will also take and return aux_data even if they do not use it.
    """

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        """
        Layer callable

        args:
            x: the input
            aux_data: data used for stuff like batch norm

        returns:
            the output MultiImage and aux_data
        """
        return x, aux_data


def get_scaled_filters(D: int, filter_block: jax.Array, weights: jax.Array) -> jax.Array:
    """
    For a set of filters and a block of weights, scale the filters by the weights.

    Consider writing a filters subclass of MultiImage that has these functions defined on it.

    args:
        D: the dimension
        filters: the geometric filters data block, (n_filters,spatial,tensor)
        weights: the block of weights, shape (out_c,in_c,n_filters)

    returns:
        array of filter sums, shape (out_c,in_c,n_filters,spatial,tensor)
    """
    _, len_k = geom.parse_shape(filter_block.shape[1:], D)
    # (out_c, in_c, n_filters) -> (out_c,in_c,n_filters,(1,)*D,(1,)*k)
    weights_mul = weights.reshape(weights.shape + (1,) * D + (1,) * len_k)

    # (out_c,in_c,n_filters,spatial,tensor)
    return filter_block[None, None] * weights_mul


def get_filter_sum(D: int, filter_block: jax.Array, weights: jax.Array) -> jax.Array:
    """
    For a set of filters and possibly a block of weights, calculate the sum of the filters scaled
    by the weights, then take the tensor norm.

    Consider writing a filters subclass of MultiImage that has these functions defined on it.

    args:
        D: dimension of the space
        filters: the geometric filters as a MultiImage
        weights: the block of weights, shape (out_c,in_c,n_filters)

    returns:
        array of filter sums, shape (out_c,in_c)
    """
    # (out_c,in_c,n_filters,spatial,tensor)
    scaled_filters = get_scaled_filters(D, filter_block, weights)

    # (out_c,in_c,tensor)
    weights_sum = jnp.sum(scaled_filters, axis=tuple(range(2, 3 + D)))

    # flatten tensor, then get its Frobenius norm.
    # (out_c,in_c,tensor_size)
    weights_sum_flat_tensor = weights_sum.reshape(weights_sum.shape[:2] + (-1,))

    # (out_c,in_c,tensor) -> (out_c,in_c)
    return jnp.linalg.norm(weights_sum_flat_tensor, axis=-1)


class AnyDimensionalModel(MultiImageModule):
    """
    A MultiImage model that implements a convertD function that can convert to work on different
    dimensional input. This also provides the helper functions transfer_weights to get this done.
    """

    @staticmethod
    def _extend_weights(
        old_weights_block: jax.Array,
        filter_key: tuple[tuple[bool, ...], int],
        old_filters: geom.MultiImage,
        new_filters: geom.MultiImage,
    ) -> jax.Array:
        """
        Given a set of weights associated with old_filters, extend the weights to new_filters.
        For offcenter weights (associated with a set of filters that has a center filter) and for
        balanced weights (associated with a set of filters which has no center filter), the new
        weights are the average of the old weights.

        args:
            old_weights_block: the old weights, shape (out_c,in_c,n_filters)
            filter_key: the key for the filters we are extending weights for
            old_filters: the old filters
            new_filters: the new filters

        returns:
            the weights associated with the new filters
        """
        k = len(filter_key[0])
        if k not in {0, 1, 2}:
            raise NotImplementedError()

        n_add_unbalanced = 0
        n_add_balanced = 0
        center_weight = None
        offcenter_old_weights = None
        balanced_weights = None
        if k == 0:
            center_weight = old_weights_block[:, :, :1]
            offcenter_old_weights = old_weights_block[:, :, 1:]
            n_add_unbalanced = len(new_filters[filter_key]) - len(old_filters[filter_key])
        elif k == 1:
            balanced_weights = old_weights_block
            n_add_balanced = len(new_filters[filter_key]) - len(old_filters[filter_key])
        elif k == 2:
            # for k==2, the first set of filters follows the scalar filters
            assert ((), 0) in old_filters, "_extend_weights needs k=0 filters if it includes k=2"
            n_old_unbalanced = len(old_filters[(), 0])
            center_weight = old_weights_block[:, :, :1]
            offcenter_old_weights = old_weights_block[:, :, 1:n_old_unbalanced]
            n_add_unbalanced = len(new_filters[(), 0]) - n_old_unbalanced

            balanced_weights = old_weights_block[:, :, n_old_unbalanced:]
            # gap between new filters and (old filters plus the additional unbalanced filter)
            n_add_balanced = len(new_filters[filter_key]) - (
                len(old_filters[filter_key]) + n_add_unbalanced
            )

        assert n_add_unbalanced >= 0
        assert n_add_balanced >= 0

        new_unbalanced_weights = jnp.zeros(old_weights_block.shape[:2] + (0,))
        if center_weight is not None and offcenter_old_weights is not None:
            # TODO: check what happens when n_add_unbalanced = 0
            additional_weights = jnp.full(
                old_weights_block.shape[:2] + (n_add_unbalanced,),
                jnp.mean(offcenter_old_weights, axis=2, keepdims=True),
            )

            new_unbalanced_weights = jnp.concatenate(
                [center_weight, offcenter_old_weights, additional_weights], axis=2
            )

        new_balanced_weights = jnp.zeros(old_weights_block.shape[:2] + (0,))
        if balanced_weights is not None:
            assert balanced_weights is not None
            additional_weights = jnp.full(
                old_weights_block.shape[:2] + (n_add_balanced,),
                jnp.mean(balanced_weights, axis=2, keepdims=True),
            )

            new_balanced_weights = jnp.concatenate([balanced_weights, additional_weights], axis=2)

        return jnp.concatenate([new_unbalanced_weights, new_balanced_weights], axis=2)

    @staticmethod
    def volume_rescale_weights(
        old_filter_triple: tuple[jax.Array, jax.Array, int],
        new_filter_triple: tuple[jax.Array, jax.Array, int],
        verbose: bool = False,
    ) -> jax.Array:
        """
        Rescale the weights so that the sum of the weights times the filters add up to the same
        value for the old filters and the new filters (which are likely a higher dimension).

        args:
            old_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the old filters, and the old dimension
            new_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the new filters, and the new dimension
            verbose: whether to print the old weights and ratios

        return:
            jax array of rescaled weights (out_channels,in_channels,num_filters) after rescaling
        """
        old_filters, old_weights, old_D = old_filter_triple
        new_filters, new_weights, new_D = new_filter_triple

        # both are (out_c,in_c)
        old_weights_sum = get_filter_sum(old_D, old_filters, old_weights)
        new_weights_sum = get_filter_sum(new_D, new_filters, new_weights)

        # Dont rescale filters that always sum to 0.
        # (n_filters,tensor)
        spatial_sum = jnp.sum(old_filters, axis=tuple(range(1, 1 + old_D)))
        # (n_filters,)
        spatial_sum_norm = jnp.linalg.norm(spatial_sum.reshape((len(spatial_sum), -1)), axis=1)
        nonzero_filter_mask = (spatial_sum_norm != 0)[None, None]  # (1,1,n_filters)

        # (out_c,in_c)
        ratios = old_weights_sum / (new_weights_sum + geom.TINY)
        # Scale nonzero by ratios, scale the others by 1 (out_c,in_c,n_filters)
        ratios = nonzero_filter_mask * ratios[..., None] + (~nonzero_filter_mask)

        if verbose:
            print("old weights", old_weights.shape, old_weights)
            print("ratios", ratios.shape, ratios)  # (out_c,in_c,n_filters)

        return new_weights * ratios

    @staticmethod
    def compatibility_norm_rescale_weights(
        old_filter_triple: tuple[jax.Array, jax.Array, int],
        new_filter_triple: tuple[jax.Array, jax.Array, int],
        verbose: bool = False,
    ) -> jax.Array:
        """
        Rescale the weight coefficients so that they are compatible with the particular embedding.
        This algorithm has an implicit assumption that we are using orthoplex filters.

        WARNING: This is the old version which works on the norms of the tensors.

        args:
            old_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the old filters, and the old dimension
            new_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the new filters, and the new dimension
            verbose: whether to print the old weights and ratios

        return:
            jax array of rescaled weights (out_channels,in_channels,num_filters) after rescaling
        """
        old_filters, old_weights, old_D = old_filter_triple
        new_filters, new_weights, new_D = new_filter_triple

        # Convert filters to the norm of the filters. This assumes 2 things:
        # 1. tensors in each pixel differ only by norm. True for nonzero filters of a single irrep
        # 2. the sign of the filters are positive
        old_filters = jnp.linalg.norm(
            old_filters.reshape(old_filters.shape[: 1 + old_D] + (-1,)), axis=-1
        )
        new_filters = jnp.linalg.norm(
            new_filters.reshape(new_filters.shape[: 1 + new_D] + (-1,)), axis=-1
        )

        # assert the filters are already in ascending order by number of pixels.
        # So for orthoplex, this means innermost to outermost
        filter_raw_sum = jnp.sum(1 * geom.nonempty_pixels(new_D, new_filters, 1), axis=-1)
        assert sorted(list(filter_raw_sum)) == list(filter_raw_sum)

        D_increase = new_D - old_D
        assert D_increase > 0

        # first, reduce the filters to the nonempty pixel filters.
        nonempty_pixel_filter = 1 * geom.nonempty_pixels(new_D, new_filters, 1).reshape(
            new_filters.shape[: 1 + new_D]
        )
        # (n_filters,old_spatial)
        collapsed_nonempty_ff = jnp.sum(nonempty_pixel_filter, axis=tuple(range(1, 1 + D_increase)))

        # (n_filters,old_spatial)
        collapsed_ff = jnp.sum(new_filters, axis=tuple(range(1, 1 + D_increase)))
        # (n_filters,old_spatial)

        # (out_c,in_c,spatial)
        old_scaled_ff = jnp.sum(get_scaled_filters(old_D, old_filters, old_weights), axis=2)

        # use np so we can easily edit it (out_c,in_c,n_nonzero_filters)
        updated_weights = np.zeros(new_weights.shape[:2] + (len(new_filters),))
        for i in reversed(range(len(filter_raw_sum))):  # starting with the outermost filter...

            # get the outermost pixel of collapsed filter i
            # (old_spatial_size,) true/falses whether the pixel is nonempty
            nonempty_pixels = geom.nonempty_pixels(old_D, collapsed_nonempty_ff[i]).ravel()
            farthest_pixel_idx = jnp.max(jnp.arange(len(nonempty_pixels))[nonempty_pixels])

            # with current weight for filter i and collapsed sum of updated_weights,
            # calculate new weight to equal old weight
            updated_weights[:, :, i] = new_weights[:, :, i]  # temp set weight to current weight
            # (out_c,in_c,n_filters,old_spatial)
            scaled_collapsed_ff = get_scaled_filters(
                old_D, collapsed_ff, jnp.array(updated_weights)
            )
            # (out_c,in_c,old_spatial)
            collapsed_sum = jnp.sum(scaled_collapsed_ff, axis=2)
            # (out_c,in_c)
            collapsed_val = collapsed_sum.reshape(collapsed_sum.shape[:2] + (-1,))[
                :, :, farthest_pixel_idx
            ]
            # assume that old_weights_val = new_weights_val. The old weight and new weight are
            # the same at this point, otherwise filter value could be different, but it wont be
            # for normalize and gaussian at least.
            old_weights_val = old_scaled_ff.reshape(collapsed_sum.shape[:2] + (-1,))[
                :, :, farthest_pixel_idx
            ]
            # this should really be new_ff_val, assume they are equal, see above
            old_norm_ff_val = old_filters[i].ravel()[farthest_pixel_idx]

            # set updated weights
            updated_weights[:, :, i] = (
                -(collapsed_val - old_weights_val) + old_weights_val
            ) / old_norm_ff_val

        updated_weights = jnp.array(updated_weights)

        # now we check that we did it right
        # (out_c,in_c,n_filters,old_spatial)
        scaled_collapsed_ff = get_scaled_filters(old_D, collapsed_ff, updated_weights)
        # (out_c,in_c,old_spatial)
        scaled_collapsed_ff = jnp.sum(scaled_collapsed_ff, axis=2)

        # (n_filters,old_spatial)
        old_norm_ff = jnp.linalg.norm(
            old_filters.reshape(old_filters.shape[: 1 + old_D] + (-1,)),
            axis=-1,
        )

        # (out_c,in_c,n_filters,old_spatial)
        old_scaled_filters = get_scaled_filters(old_D, old_norm_ff, old_weights)
        # (out_c,in_c,old_spatial)
        old_scaled_filters = jnp.sum(old_scaled_filters, axis=2)

        diff = jnp.max(jnp.abs(scaled_collapsed_ff - old_scaled_filters))
        diff_message = f"AnyDimensionalModel::compatibility_rescale_weights: Diff is {diff}"

        assert jnp.allclose(
            scaled_collapsed_ff, old_scaled_filters, rtol=1e-3, atol=1e-3
        ), diff_message

        if verbose:
            print("new weights:", new_weights)
            print("updated weights:", updated_weights)

        return updated_weights

    @staticmethod
    def compatibility_rescale_weights(
        old_filter_triple: tuple[jax.Array, jax.Array, int],
        new_filter_triple: tuple[jax.Array, jax.Array, int],
        verbose: bool = False,
    ) -> jax.Array:
        """
        Rescale the weight coefficients so that they are compatible with the particular embedding.
        This algorithm has an implicit assumption that we are using orthoplex filters. This
        implements Algorithm 1: Orthoplex filter weight scaling.

        args:
            old_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the old filters, and the old dimension
            new_filter_triple: tuple of weights (shape (out_channels,in_channels,num_filters)),
                the new filters, and the new dimension
            verbose: whether to print the old weights and ratios

        return:
            jax array of rescaled weights (out_channels,in_channels,num_filters) after rescaling
        """
        old_filters, old_weights, old_D = old_filter_triple  # old weights are alpha
        new_filters, new_weights, new_D = new_filter_triple
        k = old_filters.ndim - (1 + old_D)
        assert k == new_filters.ndim - (
            1 + new_D
        ), f"compatibility_rescale_weights: old_filters k={k}, new_filters k={new_filters.ndim - (1 + new_D)}"

        D_increase = new_D - old_D
        assert D_increase > 0, f"compatibility_rescale_weights: D_increase={D_increase}"

        # old/new_filters shape (n_filters,spatial,tensor)

        # we have filters ell=0,1,...,L
        # same number of filters
        assert len(old_filters) == len(
            new_filters
        ), f"compatibility_rescale_weights: len old_filters={len(old_filters)}, len new_filters={len(new_filters)}"
        L = len(old_filters) - 1
        L_plus = len(old_filters)  # more useful for iterating

        new_filters_proj_tensors = (
            new_filters[..., (slice(0, old_D),) * k] if k > 0 else new_filters
        )

        # currently special case N=2 because its so different
        if old_filters.shape[1] == 2 or new_filters.shape[1] == 2:
            assert (2,) * old_D == old_filters.shape[1 : 1 + old_D]
            assert (2,) * new_D == new_filters.shape[1 : 1 + new_D]

            alpha_prime = old_weights / (2**D_increase)

        else:  # filters are odd, and in particular 2L + 1 square
            # largest filter goes up to the border
            assert ((2 * L) + 1,) * old_D == old_filters.shape[1 : 1 + old_D]
            assert ((2 * L) + 1,) * new_D == new_filters.shape[1 : 1 + new_D]

            # (n_filters,new_spatial)
            new_filters_proj_norm = jnp.linalg.norm(
                new_filters_proj_tensors.reshape(new_filters.shape[: 1 + new_D] + (-1,)), axis=-1
            )

            # (n_filters,old_spatial)
            old_filters_norm = jnp.linalg.norm(
                old_filters.reshape(old_filters.shape[: 1 + old_D] + (-1,)), axis=-1
            )

            # use np so we can easily edit it (out_c,in_c,n_nonzero_filters)
            alpha_prime = np.zeros(new_weights.shape[:2] + (L_plus,))
            for z in reversed(range(L_plus)):  # iterates from L,L-1,...,0
                j_d_centered = (z,) + (0,) * (old_D - 1)
                j_dplus_centered = (z,) + (0,) * (new_D - 1)

                j_d = tuple(x + L for x in j_d_centered)
                j_dplus = tuple(x + L for x in j_dplus_centered)

                # (out_c,in_c,n_filters,new_spatial)
                scaled_new_filters = (
                    alpha_prime[..., *((None,) * new_D)] * new_filters_proj_norm[None, None]
                )
                # sum over filters, spatial dims (out_c,in_c,old_spatial)
                # since alpha_prime are only nonzero for z+1, this is the proper sum over ell=z+1 to L
                collapsed_ff = jnp.sum(scaled_new_filters, axis=tuple(range(2, 2 + 1 + D_increase)))

                # alpha_prime = (alpha * C_z - sum) / (C'_z)
                alpha_prime[:, :, z] = (
                    old_weights[:, :, z] * old_filters_norm[z, *j_d] - collapsed_ff[:, :, *j_d]
                ) / new_filters_proj_norm[z, *j_dplus]

            alpha_prime = jnp.array(alpha_prime)

        # now we check that we did it right
        # (out_c,in_c,n_filters,new_spatial,proj_tensor)
        scaled_new_filters = (
            alpha_prime[..., *((None,) * (new_D + k))] * new_filters_proj_tensors[None, None]
        )
        # (out_c,in_c,old_spatial,proj_tensor)
        collapsed_ff = jnp.sum(scaled_new_filters, axis=tuple(range(2, 2 + 1 + D_increase)))

        # (out_c,in_c,n_filters,old_spatial,tensor)
        scaled_old_filters = old_weights[..., *((None,) * (old_D + k))] * old_filters[None, None]
        # (out_c,in_c,old_spatial,tensor)
        scaled_old_filters = jnp.sum(scaled_old_filters, axis=2)

        diff = jnp.max(jnp.abs(collapsed_ff - scaled_old_filters))
        diff_message = f"AnyDimensionalModel::compatibility_rescale_weights: Diff is {diff}"

        assert jnp.allclose(collapsed_ff, scaled_old_filters, rtol=1e-3, atol=1e-3), diff_message

        if verbose:
            print("new weights:", new_weights)
            print("updated weights:", alpha_prime)

        return alpha_prime

    @staticmethod
    def _transfer_conv_weights(
        weights: dict[tuple[tuple[bool, ...], int], dict[tuple[tuple[bool, ...], int], jax.Array]],
        old_filters: geom.MultiImage,
        new_filters: geom.MultiImage,
        rescale: geom.Rescaling,
        verbose: bool = False,
    ) -> dict[tuple[tuple[bool, ...], int], dict[tuple[tuple[bool, ...], int], jax.Array]]:
        """
        Transfer the conv weights from old filters to new filters of possibly a different dimension.
        If rescale is true, then scale the weights so that the sum of the filter basis of a particular
        order scaled by the weights is equal for the old filters and the new.

        args:
            weights: a weights dictionary from a layers.ConvContract layer
            old_filters: the old filters that the weights came from
            new_filters: the new filters that we will be using the weights for
            rescale: type of rescaling to perform on the weights
            verbose: print the ratio of the squared sum of filters new/old after transfering the
                weights, default to False.

        returns:
            a new weights dictionary
        """
        new_weights = {}

        for (in_k, in_p), in_weights in weights.items():
            new_weights[(in_k, in_p)] = {}
            for (out_k, out_p), old_weights_block in in_weights.items():
                filter_k = in_k + out_k
                filter_key = (filter_k, (in_p + out_p) % 2)

                new_weights_block = AnyDimensionalModel._extend_weights(
                    old_weights_block, filter_key, old_filters, new_filters
                )

                old_filter_block = old_filters[filter_key]
                new_filter_block = new_filters[filter_key]

                if rescale is geom.Rescaling.VOLUME:
                    pos_weights = AnyDimensionalModel.volume_rescale_weights(
                        (old_filter_block, jax.nn.relu(old_weights_block), old_filters.D),
                        (new_filter_block, jax.nn.relu(new_weights_block), new_filters.D),
                        verbose,
                    )
                    neg_weights = AnyDimensionalModel.volume_rescale_weights(
                        (old_filter_block, -jax.nn.relu(-old_weights_block), old_filters.D),
                        (new_filter_block, -jax.nn.relu(-new_weights_block), new_filters.D),
                        verbose,
                    )
                    scaled_weights_block = pos_weights + neg_weights
                elif rescale is geom.Rescaling.COMPATIBILITY:
                    # Dont rescale filters that always sum to 0.
                    # (n_filters,tensor)
                    spatial_sum = jnp.sum(new_filter_block, axis=tuple(range(1, 1 + new_filters.D)))
                    # (n_filters,)
                    spatial_sum_norm = jnp.linalg.norm(
                        spatial_sum.reshape((len(spatial_sum), -1)), axis=1
                    )
                    nonzero_mask = spatial_sum_norm != 0  # (n_filters,)

                    updated_weights_block = AnyDimensionalModel.compatibility_rescale_weights(
                        (
                            old_filter_block[nonzero_mask],
                            old_weights_block[:, :, nonzero_mask],
                            old_filters.D,
                        ),
                        (
                            new_filter_block[nonzero_mask],
                            new_weights_block[:, :, nonzero_mask],
                            new_filters.D,
                        ),
                        verbose,
                    )

                    scaled_weights_block = new_weights_block
                    scaled_weights_block = scaled_weights_block.at[:, :, nonzero_mask].set(
                        updated_weights_block
                    )
                else:
                    scaled_weights_block = new_weights_block

                new_weights[(in_k, in_p)][(out_k, out_p)] = scaled_weights_block

        return new_weights

    def transfer_weights(
        self: Self, new_model: Self, rescale: geom.Rescaling, verbose: bool = False
    ) -> Self:
        """
        Transfer the weights and biases from an old model to a new model. This allows converting
        between dimensions as well. This works by copying all jax arrays from the old model to the new
        model, then resetting the new models conv filters to the new conv filters, then doing any
        conv filter related weight scaling.

        In the future, it may make sense for the updates to be defined on the individual layers, and
        then the tree_at recursively calls those functions.

        args:
            old_model: the old model
            new_model: the new model
            old_conv_filters: the convolution filters used in the old model
            conv_filters: the convolution filters to use in the new model, can have different D
            rescale: type of rescaling to perform on the weights
            verbose: print the ratio of the squared sum of filters new/old after transfering the
                weights, default to False.

        returns:
            a new model with the old weights except conv weights which are adjusted, and new filters
        """
        # get the new filters
        is_conv = lambda n: isinstance(n, layers.ConvContract)
        get_filters = lambda m: [
            x.invariant_filters for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
        ]
        new_filters = get_filters(new_model)

        # now replace all jax arrays
        get_all_weights = lambda m: jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array)
        new_model = eqx.tree_at(get_all_weights, new_model, get_all_weights(self))

        # now reset the proper conv filters
        new_model = eqx.tree_at(get_filters, new_model, new_filters)

        # now set the proper weights
        get_conv_weights = lambda m: [
            x.weights for x in jax.tree_util.tree_leaves(m, is_leaf=is_conv) if is_conv(x)
        ]
        conv_weights = get_conv_weights(self)
        new_weights = [
            AnyDimensionalModel._transfer_conv_weights(
                weight, old_filter, new_filter, rescale, verbose
            )
            for weight, old_filter, new_filter in zip(conv_weights, get_filters(self), new_filters)
        ]
        new_model = eqx.tree_at(get_conv_weights, new_model, new_weights)

        return new_model

    def convertD(
        self: Self, conv_filters: geom.MultiImage, rescale: geom.Rescaling, key: jax.Array, **kwargs
    ) -> Self:
        """
        Placeholder function, must be overwritten by the inheriting class.

        Construct a new model with filters in a higher dimension. This only works for equivariant
        models.

        args:
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: type of rescaling to perform on the weights
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        raise NotImplementedError(
            f"AnyDimensionalModel::convertD: derived class {self.__class__} does not implement convertD."
        )


class ConvBlock(MultiImageModule):
    """
    A convolution block consisting of a convolution, a nonlinearity, and a GroupNorm/BatchNorm.
    Can be equivariant or not, in typical order or in preactivation order.
    """

    conv: layers.ConvContract | layers.LayerWrapper
    group_norm: layers.GroupNorm | layers.LayerWrapper | None
    batch_norm: layers.LayerWrapperAux | None
    nonlinearity: layers.VectorNeuronNonlinear | layers.LayerWrapper | Callable

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
                self.group_norm = layers.LayerNorm(output_keys, self.D)
            else:
                self.group_norm = layers.LayerWrapper(
                    eqx.nn.GroupNorm(1, output_keys[0][1]), output_keys
                )
        else:
            self.group_norm = None

        if use_batch_norm:
            self.batch_norm = layers.LayerWrapperAux(
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


class UNet(AnyDimensionalModel):
    """
    Implementation of the UNet: https://arxiv.org/abs/1505.04597.
    This model defaults to the equivariant version, but can also be the non-equivariant version.
    """

    embedding: list[ConvBlock]
    downsample_blocks: list[tuple[layers.MaxNormPool, list[ConvBlock]]]
    upsample_blocks: list[tuple[layers.ConvContract | layers.LayerWrapper, list[ConvBlock]]]
    decode: layers.ConvContract | layers.LayerWrapper

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)
    activation_f: Callable | str | None = eqx.field(static=True)
    use_group_norm: bool = eqx.field(static=True)
    use_batch_norm: bool = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)
    mid_keys: geom.Signature = eqx.field(static=True)
    padding_mode: str = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_downsamples: int = 4,
        num_conv: int = 2,
        use_bias: Union[bool, str] = "auto",
        activation_f: Callable | str | None = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: Optional[geom.MultiImage] = None,
        upsample_filters: Optional[geom.MultiImage] = None,
        kernel_size: Optional[Union[int, Sequence[int]]] = None,
        use_group_norm: bool = False,
        use_batch_norm: bool = False,
        mid_keys: Optional[geom.Signature] = None,
        padding_mode: str = "ZEROS",
        key: Any = None,
    ) -> None:
        """
        Constructor for the UNet.

        args:
            D: the dimension of the space
            input_keys: the MultiImage Signature for the input
            output_keys: the MultiImage Signature for the output
            depth: the number of channels at the highest level of the unet. This is overwritten if
                mid_keys is provided
            num_downsamples: number of convolution blocks followed by a max pool
            num_conv: number of convolutions per level
            use_bias: whether to use a bias
            activation_f: the activation function
            equivariant: whether to be equivariant
            conv_filters: the invariant filters for the equivariant version
            kernel_size: sidelength(s) for the non-equivariant version
            use_group_norm: whether to use GroupNorm
            use_batch_norm: whether to use the BatchNorm, only for non-equivariant version
            mid_keys: types of images and number of channels for the mid layers, as a baseline
            padding_mode: used for non-equivariant models, padding mode to pass to convolutions
            key: jax.random key
        """
        assert num_conv > 0
        assert key is not None

        self.input_keys = input_keys
        self.output_keys = output_keys
        if equivariant:
            if mid_keys is None:
                mid_keys = geom.signature_union(input_keys, output_keys, depth)

            assert not use_batch_norm, "UNet::init Batch Norm cannot be used with equivariant model"
        else:
            if mid_keys is None:
                mid_keys = geom.Signature(((((), 0), depth),))

            # use these keys along the way, then for the final output use self.output_keys
            input_keys_size = sum(in_c * (D ** len(k)) for (k, _), in_c in input_keys)
            input_keys = geom.Signature(((((), 0), input_keys_size),))
            output_key_size = sum(out_c * (D ** len(k)) for (k, _), out_c in output_keys)
            output_keys = geom.Signature(((((), 0), output_key_size),))

        self.D = D
        self.equivariant = equivariant
        self.use_bias = use_bias
        self.activation_f = activation_f
        self.use_group_norm = use_group_norm
        self.use_batch_norm = use_batch_norm
        self.mid_keys = mid_keys
        self.padding_mode = padding_mode

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
                    padding_mode=padding_mode,
                    key=subkey,
                )
            )

        self.downsample_blocks = []
        for downsample in range(1, num_downsamples + 1):
            down_layers = (layers.MaxNormPool(2, equivariant), [])

            for conv_idx in range(num_conv):
                out_keys = geom.Signature(
                    tuple((k_p, _depth * (2**downsample)) for k_p, _depth in mid_keys)
                )
                if conv_idx == 0:
                    in_keys = geom.Signature(
                        tuple((k_p, _depth * (2 ** (downsample - 1))) for k_p, _depth in mid_keys)
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
                        padding_mode=padding_mode,
                        key=subkey,
                    )
                )

            self.downsample_blocks.append(down_layers)

        self.upsample_blocks = []
        for upsample in reversed(range(num_downsamples)):
            in_keys = geom.Signature(
                tuple((k_p, _depth * (2 ** (upsample + 1))) for k_p, _depth in mid_keys)
            )
            out_keys = geom.Signature(
                tuple((k_p, _depth * (2**upsample)) for k_p, _depth in mid_keys)
            )
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
                    padding_mode=padding_mode,
                    key=subkey,
                ),
                [],
            )

            for conv_idx in range(num_conv):
                out_keys = geom.Signature(
                    tuple((k_p, _depth * (2**upsample)) for k_p, _depth in mid_keys)
                )
                if conv_idx == 0:  # due to adding the residual layer back, in_c is doubled again
                    in_keys = geom.Signature(
                        tuple((k_p, _depth * (2 ** (upsample + 1))) for k_p, _depth in mid_keys)
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
                        padding_mode=padding_mode,
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
            padding_mode=padding_mode,
            key=subkey,
        )

    def convertD(
        self: Self,
        conv_filters: geom.MultiImage,
        rescale: geom.Rescaling,
        key: jax.Array,
        **kwargs,
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension. This only works for equivariant
        models.

        args:
            old_conv_filters: the current conv filters for the model
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        assert self.equivariant
        assert "upsample_filters" in kwargs
        new_model = self.__class__(
            conv_filters.D,
            self.input_keys,
            self.output_keys,
            0,  # ignored since mid_keys is provided
            len(self.downsample_blocks),
            len(self.embedding),
            self.use_bias,
            self.activation_f,
            self.equivariant,
            conv_filters,
            kwargs["upsample_filters"],
            0,  # ignored for equivariant model
            self.use_group_norm,
            self.use_batch_norm,
            self.mid_keys,
            self.padding_mode,
            key,
        )

        return self.transfer_weights(new_model, rescale)

    def __call__(
        self: Self, x: geom.MultiImage, batch_stats: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
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

        return out, batch_stats


class DilResNet(AnyDimensionalModel):
    """
    The Dilated ResNet from https://arxiv.org/abs/2112.15275.
    """

    encoder: list[ConvBlock]
    blocks: list[list[ConvBlock]]
    decoder: list[ConvBlock]

    D: int = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)
    activation_f: Callable | str | None = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    use_group_norm: bool = eqx.field(static=True)
    mid_keys: geom.Signature = eqx.field(static=True)
    padding_mode: str = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_blocks: int = 4,
        use_bias: bool | str = "auto",
        activation_f: Callable | str | None = jax.nn.relu,
        equivariant: bool = True,
        conv_filters: geom.MultiImage | None = None,
        kernel_size: int | Sequence[int] | None = None,
        use_group_norm: bool = False,
        mid_keys: geom.Signature | None = None,
        padding_mode: str = "ZEROS",
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
            mid_keys: types of images and number of channels for the mid layers, as a baseline
            padding_mode: used for non-equivariant models, padding mode to pass to convolutions
            key: jax.random key
        """
        self.D = D
        self.equivariant = equivariant
        self.output_keys = output_keys
        self.input_keys = input_keys

        if equivariant:
            if mid_keys is None:
                mid_keys = geom.signature_union(input_keys, output_keys, depth)
        else:
            if mid_keys is None:
                mid_keys = geom.Signature(((((), 0), depth),))

            # use these keys along the way, then for the final output use self.output_keys
            input_keys = geom.Signature(
                ((((), 0), sum(in_c * (D ** len(k)) for (k, _), in_c in input_keys)),)
            )
            output_keys = geom.Signature(
                ((((), 0), sum(out_c * (D ** len(k)) for (k, _), out_c in output_keys)),)
            )

        self.use_bias = use_bias
        self.activation_f = activation_f
        self.use_group_norm = use_group_norm
        self.mid_keys = mid_keys
        self.padding_mode = padding_mode

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
                padding_mode=padding_mode,
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
                padding_mode=padding_mode,
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
                        padding_mode=padding_mode,
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
                padding_mode=padding_mode,
                key=subkey1,
            ),
            ConvBlock(
                D,
                mid_keys,
                output_keys,
                use_bias,
                None,
                equivariant,
                conv_filters,
                1,
                padding_mode=padding_mode,
                key=subkey2,
            ),
        ]

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        """
        Callable for this layer

        args:
            x: the input MultiImage
            aux_data: unused, needed for compliance

        returns:
            the output MultiImage, aux_data
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

        return out, aux_data

    def convertD(
        self: Self,
        conv_filters: geom.MultiImage,
        rescale: geom.Rescaling,
        key: jax.Array,
        **kwargs,
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension. This only works for equivariant
        models.

        args:
            old_conv_filters: the current conv filters for the model
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        assert self.equivariant

        new_model = self.__class__(
            conv_filters.D,
            self.input_keys,
            self.output_keys,
            0,  # ignored since mid_keys is provided
            len(self.blocks),
            self.use_bias,
            self.activation_f,
            self.equivariant,
            conv_filters,
            0,  # ignored for equivariant model
            self.use_group_norm,
            self.mid_keys,
            self.padding_mode,
            key,
        )

        return self.transfer_weights(new_model, rescale)


class ResNet(AnyDimensionalModel):
    """
    A typical ResNet.
    """

    encoder: list[ConvBlock]
    blocks: list[list[ConvBlock]]
    decoder: list[ConvBlock]

    D: int = eqx.field(static=True)
    equivariant: bool = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)
    activation_f: Callable | str = eqx.field(static=True)
    use_group_norm: bool = eqx.field(static=True)
    preactivation_order: bool = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)
    mid_keys: geom.Signature = eqx.field(static=True)
    padding_mode: str = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        input_keys: geom.Signature,
        output_keys: geom.Signature,
        depth: int,
        num_blocks: int = 8,
        num_conv: int = 2,
        use_bias: bool | str = "auto",
        activation_f: Callable | str = jax.nn.gelu,
        equivariant: bool = True,
        conv_filters: geom.MultiImage | None = None,
        kernel_size: int | Sequence[int] | None = None,
        use_group_norm: bool = True,
        preactivation_order: bool = True,
        mid_keys: geom.Signature | None = None,
        padding_mode: str = "ZEROS",
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
            mid_keys: types of images and number of channels for the mid layers, as a baseline
            padding_mode: for non-equivariant, pass 'TOROIDAL' if all sides are toroidal
            key: jax.random key
        """
        self.D = D
        self.equivariant = equivariant
        self.output_keys = output_keys
        self.input_keys = input_keys

        if equivariant:
            if mid_keys is None:
                mid_keys = geom.signature_union(input_keys, output_keys, depth)
        else:
            if mid_keys is None:
                mid_keys = geom.Signature(((((), 0), depth),))

            # use these keys along the way, then for the final output use self.output_keys
            input_keys = geom.Signature(
                ((((), 0), sum(in_c * (D ** len(k)) for (k, _), in_c in input_keys)),)
            )
            output_keys = geom.Signature(
                ((((), 0), sum(out_c * (D ** len(k)) for (k, _), out_c in output_keys)),)
            )

        self.use_bias = use_bias
        self.activation_f = activation_f
        self.use_group_norm = use_group_norm
        self.preactivation_order = preactivation_order
        self.mid_keys = mid_keys
        self.padding_mode = padding_mode

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
                padding_mode=padding_mode,
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
                padding_mode=padding_mode,
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
                        padding_mode=padding_mode,
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
                padding_mode=padding_mode,
                key=subkey1,
            ),
            ConvBlock(
                D,
                mid_keys,
                output_keys,
                use_bias,
                None,
                equivariant,
                conv_filters,
                1,
                padding_mode=padding_mode,
                key=subkey2,
            ),
        ]

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        """
        Callable for this layer

        args:
            x: the input MultiImage
            aux_data: unused, needed for compliance

        returns:
            the output MultiImage and aux_data
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

        return out, aux_data

    def convertD(
        self: Self,
        conv_filters: geom.MultiImage,
        rescale: geom.Rescaling,
        key: jax.Array,
        **kwargs,
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension. This only works for equivariant
        models.

        args:
            old_conv_filters: the current conv filters for the model
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        assert self.equivariant

        new_model = self.__class__(
            conv_filters.D,
            self.input_keys,
            self.output_keys,
            0,  # ignored since mid_keys is provided
            len(self.blocks),
            len(self.blocks[0]),
            self.use_bias,
            self.activation_f,
            self.equivariant,
            conv_filters,
            0,  # ignored for equivariant model
            self.use_group_norm,
            self.preactivation_order,
            self.mid_keys,
            self.padding_mode,
            key,
        )

        return self.transfer_weights(new_model, rescale)


class ModelWrapper(MultiImageModule):
    """
    This wraps a typical CNN so that it is a MultiImage model. This model will take an input
    MultiImage, convert it to a jax array, feed it through the model, then convert it to the
    appropriate output MultiImage at the end.
    """

    model: eqx.Module

    D: int = eqx.field(static=True)
    output_keys: geom.Signature = eqx.field(static=True)
    output_is_torus: Union[bool, tuple[bool, ...]] = eqx.field(static=True)
    pass_aux_data: bool = eqx.field(static=True)

    def __init__(
        self: Self,
        D: int,
        model: eqx.Module,
        output_keys: geom.Signature,
        output_is_torus: Union[bool, tuple[bool, ...]],
        pass_aux_data: bool = False,
    ) -> None:
        """
        Construct the model wrapper.

        args:
            D: the dimension of the space
            model: a vanilla cnn model, should input and output images of shape (channels,spatial)
            output_keys: signature for the output MultiImage
            output_is_torus: toroidal structure of the output MultiImage
            pass_aux_data: whether the model expects and outputs aux_data
        """
        self.D = D
        assert callable(model)
        self.model = model
        self.output_keys = output_keys
        self.output_is_torus = output_is_torus
        self.pass_aux_data = pass_aux_data  # pass the AUX, bro

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        x_array = x.to_scalar_multi_image()[((), 0)]
        assert callable(self.model)
        if self.pass_aux_data:
            out, aux_data = self.model(x_array, aux_data)
        else:
            out = self.model(x_array)

        out_multi_image = geom.MultiImage(
            {(0, 0): out},
            self.D,
            self.output_is_torus,
        ).from_scalar_multi_image(self.output_keys)

        return out_multi_image, aux_data


class GroupAverage(MultiImageModule):
    """
    Model that takes in a different model and peforms group averaging to make it an equivariant
    model. Can either always average, so that it is equivariant during training as well, or only
    average at inference time to test whether training a non-equivariant model, then group
    averaging helps. This will reveal whether to data set is indeed an equivariant data set.
    """

    model: MultiImageModule
    inference: bool

    # static to prevent this from being converted to a traced jax array
    operators: list[np.ndarray] = eqx.field(static=True)
    always_average: bool = eqx.field(static=True)

    def __init__(
        self: Self,
        model: MultiImageModule,
        operators: list[np.ndarray],
        always_average: bool = False,
        inference: bool = False,
    ) -> None:
        self.model = model
        self.operators = operators
        self.always_average = always_average
        self.inference = inference

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:

        if (self.always_average or self.inference) and len(self.operators) > 0:
            sum_image = None
            out_aux = None
            for gg in self.operators:
                out_image, out_aux = self.model(x.times_group_element(gg), aux_data)
                rot_out_image = out_image.times_group_element(gg.T)
                sum_image = rot_out_image if sum_image is None else sum_image + rot_out_image

            assert sum_image is not None
            return sum_image / len(self.operators), out_aux

        else:
            return self.model(x, aux_data)


class Climate1D(MultiImageModule):

    model: MultiImageModule

    output_keys: geom.Signature = eqx.field(static=True)
    past_steps: int = eqx.field(static=True)
    future_steps: int = eqx.field(static=True)
    spatial_dims: tuple[int, ...] = eqx.field(static=True)
    constant_fields_2d: dict[tuple[tuple[bool, ...], int], int] = eqx.field(static=True)
    output_is_torus: tuple[bool, ...] = eqx.field(static=True)

    def __init__(
        self: Self,
        model: MultiImageModule,
        output_keys: geom.Signature,
        past_steps: int,
        future_steps: int,
        spatial_dims: tuple[int, ...],
        constant_fields_2d: dict[tuple[tuple[bool, ...], int], int],
        output_is_torus: tuple[bool, ...] = (True, False),
    ) -> None:
        self.model = model
        self.output_keys = output_keys
        self.past_steps = past_steps
        self.future_steps = future_steps
        self.spatial_dims = spatial_dims  # 2d
        self.constant_fields_2d = constant_fields_2d
        self.output_is_torus = output_is_torus

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State] = None
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        assert aux_data is None, "Currently cannot handle batch stats"

        # we multiply this by the identity
        x1 = self.from1d(self.model(self.to1d(x), aux_data)[0])

        equator_flip = np.array([[1, 0], [0, -1]])
        x2 = self.from1d(
            self.model(self.to1d(x.times_group_element(equator_flip)), aux_data)[0]
        ).times_group_element(equator_flip)

        return (x1 + x2) / 2, aux_data

    def to1d(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        spatial_dims = x.get_spatial_dims()
        n_lons, _ = spatial_dims

        dynamic_x, const_x = x.concat_inverse(self.constant_fields_2d)
        dynamic_x = dynamic_x.expand(0, self.past_steps)

        out = geom.MultiImage({}, 1, (True,))
        for (k, parity), image in dynamic_x.items():
            assert (k, parity) in [((), 0), ((), 1), ((False,), 0)]  # currently must be one of

            if k == ():
                out.append(k, parity, image)
            else:  # k==1
                # velocity in horizontal direction becomes a pseudoscalar, vertical is a scalar
                out.append(0, 1, image[..., 0])
                out.append(0, 0, image[..., 1])

        # (c,t,x,y) -> (y,c,t,x) -> (y*c*t,x)
        out.data = {
            (k, parity): jnp.moveaxis(image, -1, 0).reshape((-1, n_lons))
            for (k, parity), image in out.items()
        }

        for (k, parity), image in const_x.items():
            # (c,x,y) -> (y,c,x) -> (y*c,x)
            out.append(k, parity, jnp.moveaxis(image, -1, 0).reshape((-1, n_lons)))

        return out

    def from1d(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        n_lons, n_lats = self.spatial_dims
        keys_dict = {(k, parity): size for (k, parity), size in self.output_keys}

        # number of channels
        c_scalar = keys_dict[((), 0)] // self.future_steps if ((), 0) in keys_dict else 0
        c_pseudoscalar = keys_dict[((), 1)] // self.future_steps if ((), 1) in keys_dict else 0
        c_vector = (
            keys_dict[((False,), 0)] // self.future_steps if ((False,), 0) in keys_dict else 0
        )
        # does this need to be able to handle covariant axes

        out = geom.MultiImage({}, 2, self.output_is_torus)
        x = x.expand(0, self.future_steps)  # -> (y*c,t,x)

        scalar_image = None
        pseudoscalar_image = None
        if ((), 0) in x:
            # (y*c,t,x) -> (y,c,t,x) -> (c,t,x,y)
            scalar_image = jnp.moveaxis(
                x[((), 0)].reshape((n_lats, -1, self.future_steps, n_lons)), 0, -1
            )
            assert len(scalar_image) == c_scalar + c_vector
        if ((), 1) in x:
            # (y*c,t,x) -> (y,c,t,x) -> (c,t,x,y)
            pseudoscalar_image = jnp.moveaxis(
                x[((), 1)].reshape((n_lats, -1, self.future_steps, n_lons)), 0, -1
            )
            assert len(pseudoscalar_image) == c_pseudoscalar + c_vector

        vec = None
        if ((False,), 0) in keys_dict:  # then there are scalars and pseudoscalars
            assert scalar_image is not None and pseudoscalar_image is not None
            vec_y = scalar_image[c_scalar:]
            scalar_image = scalar_image[:c_scalar]

            vec_x = pseudoscalar_image[c_pseudoscalar:]
            pseudoscalar_image = pseudoscalar_image[:c_pseudoscalar]
            vec = jnp.stack([vec_x, vec_y], axis=-1)

        if ((), 0) in keys_dict:
            assert scalar_image is not None
            out.append(0, 0, scalar_image)
        if ((), 1) in keys_dict:
            assert pseudoscalar_image is not None
            out.append(0, 1, pseudoscalar_image)
        if ((False,), 0) in keys_dict:
            assert vec is not None
            out.append((False,), 0, vec)

        return out.combine_axes((0, 1))

    @classmethod
    def get_1d_signature(
        cls, signature: Union[geom.Signature, dict[tuple[tuple[bool, ...], int], int]], n_lats: int
    ) -> geom.Signature:
        if not isinstance(signature, dict):
            signature = {(k, parity): size for (k, parity), size in signature}

        new_signature = {}
        for (k, parity), size in signature.items():
            assert (k, parity) in [((), 0), ((), 1), ((False,), 0)]

            if k == ():
                if (k, parity) not in new_signature:
                    new_signature[(k, parity)] = size * n_lats
                else:
                    new_signature[(k, parity)] += size * n_lats
            else:  # k ==1
                if ((), 0) not in new_signature:
                    new_signature[((), 0)] = size * n_lats
                else:
                    new_signature[((), 0)] += size * n_lats

                if ((), 1) not in new_signature:
                    new_signature[((), 1)] = size * n_lats
                else:
                    new_signature[((), 1)] += size * n_lats

        return geom.Signature(tuple(new_signature.items()))


class LastStepIdentity(AnyDimensionalModel):

    residual: bool = eqx.field(static=True)

    def __init__(self: Self, residual: bool = False):
        self.residual = residual

    def convertD(
        self: Self, conv_filters: geom.MultiImage, rescale: geom.Rescaling, key: jax.Array, **kwargs
    ) -> Self:
        """
        Convert model to a different dimension.

        args:
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        return self.__class__(self.residual)

    def __call__(
        self: Self, x: geom.MultiImage, batch_stats: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        """
        Callable function.

        args:
            x: the input MultiImage
            batch_stats: batch stats for BatchNorm if present

        returns:
            the output MultiImage and batch_stats
        """

        out = x.empty()
        for (k, parity), img_block in x.items():
            # If it is a residual model, make it all zeros to add it
            out_img_block = jnp.zeros_like(img_block[-1:]) if self.residual else img_block[-1:]
            out.append(k, parity, out_img_block)

        return out, batch_stats


class SimpleConvSeries(AnyDimensionalModel):
    """
    Simple convolution model consisting of a series of ConvBlocks, with all but the last with a
    gelu vector neuron nonlinearity.
    """

    layers: list[ConvBlock]

    D: int = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    width: int = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        width: int,
        depth: int,
        use_bias: bool | str,
        key: jax.Array,
    ) -> None:
        assert depth >= 1
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.width = width
        self.depth = depth
        self.use_bias = use_bias

        mid_keys = geom.signature_union(input_keys, target_keys, width) if depth > 1 else input_keys

        subkey_last, *subkeys = random.split(key, num=depth)
        self.layers = []
        for subkey in subkeys:
            self.layers.append(
                ConvBlock(
                    self.D, input_keys, mid_keys, use_bias, "gelu", True, conv_filters, key=subkey
                )
            )

        self.layers.append(
            ConvBlock(
                self.D, mid_keys, target_keys, use_bias, None, True, conv_filters, key=subkey_last
            )
        )

    def convertD(
        self: Self, conv_filters: geom.MultiImage, rescale: geom.Rescaling, key: jax.Array, **kwargs
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension.

        args:
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: how to rescale the filter weights
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        new_model = self.__class__(
            self.input_keys,
            self.target_keys,
            conv_filters,
            self.width,
            self.depth,
            self.use_bias,
            key,
        )

        return self.transfer_weights(new_model, rescale, verbose=False)

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        for layer in self.layers:
            x, aux_data = layer(x, aux_data)

        return x, aux_data
