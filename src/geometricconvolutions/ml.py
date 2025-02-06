import time
import math
import functools
from typing import Any, Callable, NewType, Optional, Sequence, Union
from typing_extensions import Self
import scipy.special
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike
import equinox as eqx
import optax

import geometricconvolutions.geometric as geom

LayerKey = NewType("LayerKey", tuple[int, int])


# ~~~~~~~~~~~~~~~~~~~~~~ Layers ~~~~~~~~~~~~~~~~~~~~~~
class ConvContract(eqx.Module):
    weights: dict[LayerKey, dict[LayerKey, jax.Array]]
    bias: dict[LayerKey, jax.Array]
    invariant_filters: geom.Layer

    input_keys: tuple[tuple[LayerKey, int]] = eqx.field(static=True)
    target_keys: tuple[tuple[LayerKey, int]] = eqx.field(static=True)
    use_bias: Union[str, bool] = eqx.field(static=True)
    stride: Optional[tuple[int]] = eqx.field(static=True)
    padding: Optional[tuple[int]] = eqx.field(static=True)
    lhs_dilation: Optional[tuple[int]] = eqx.field(static=True)
    rhs_dilation: Optional[tuple[int]] = eqx.field(static=True)
    D: int = eqx.field(static=True)
    fast_mode: bool = eqx.field(static=True)
    missing_filter: bool = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: tuple[tuple[LayerKey, int]],
        target_keys: tuple[tuple[LayerKey, int]],
        invariant_filters: geom.Layer,
        use_bias: Union[str, bool] = "auto",
        stride: Optional[tuple[int]] = None,
        padding: Optional[tuple[int]] = None,
        lhs_dilation: Optional[tuple[int]] = None,
        rhs_dilation: Optional[tuple[int]] = None,
        key: Optional[ArrayLike] = None,
    ):
        """
        Equivariant tensor convolution then contraction.
        args:
            input_keys: A mapping of (k,p) to an integer representing the input channels
            target_keys: A mapping of (k,p) to an integer representing the output channels
            invariant_filters: A Layer of the invariant filters to build the convolution filters
            use_bias: One of 'auto', 'mean', or 'scalar', or True for 'auto' or False for no bias.
                Mean uses a mean scale for every type, scalar uses a regular bias for scalars only
                and auto does regular bias for scalars and mean for non-scalars. Defaults to auto.
            For the rest of arguments, see convolve
        """
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.invariant_filters = invariant_filters
        self.use_bias = use_bias
        self.stride = stride
        self.padding = padding
        self.lhs_dilation = lhs_dilation
        self.rhs_dilation = rhs_dilation

        self.D = invariant_filters.D
        # if a particular desired convolution for input_keys -> target_keys is missing the needed
        # filter (possibly because an equivariant one doesn't exist), this is set to true
        self.missing_filter = False

        if isinstance(use_bias, bool):
            use_bias = "auto" if use_bias else use_bias
        elif isinstance(use_bias, str):
            assert use_bias in {"auto", "mean", "scalar"}
        else:
            raise ValueError(
                f"ConvContract: bias must be str or bool, but found {type(use_bias)}:{use_bias}"
            )

        self.weights = {}  # presumably some way to jax.lax.scan this?
        self.bias = {}
        all_filter_spatial_dims = []
        for (in_k, in_p), in_c in self.input_keys:
            self.weights[(in_k, in_p)] = {}
            for (out_k, out_p), out_c in self.target_keys:
                key, subkey1, subkey2 = random.split(key, num=3)

                filter_key = (in_k + out_k, (in_p + out_p) % 2)
                if filter_key not in self.invariant_filters:
                    self.missing_filter = True
                    continue  # relevant when there isn't an N=3, (0,1) filter

                num_filters = len(self.invariant_filters[filter_key])
                if False and filter_key == (0, 0):
                    # TODO: Currently unused, a work in progress
                    weight_per_ff = []
                    # TODO: jax.lax.scan here instead
                    for conv_filter, tensor_mul in zip(
                        self.invariant_filters[filter_key],
                        [1, (1 + 8 / 9), (1 + 2 / 3)],
                        # [1, 1, 1],
                    ):
                        key, subkey = random.split(key)

                        # number of weights that will appear in a single component output.
                        tensor_mul = scipy.special.comb(jnp.sum(conv_filter), 2, repetition=True)
                        # tensor_mul = jnp.sum(conv_filter**2, axis=tuple(range(self.D))) * tensor_mul
                        bound = jnp.sqrt(1 / (in_c * num_filters * tensor_mul))

                        weight_per_ff.append(
                            random.uniform(subkey, shape=(out_c, in_c), minval=-bound, maxval=bound)
                        )
                    self.weights[(in_k, in_p)][(out_k, out_p)] = jnp.stack(weight_per_ff, axis=-1)

                    # # bound = jnp.sqrt(3 / (0.085 * in_c * num_filters)) # tanh multiplier
                    # bound = jnp.sqrt(3 / (in_c * num_filters))
                    # key, subkey = random.split(key)
                    # rand_weights = random.uniform(
                    #     subkey, shape=(out_c, in_c, num_filters), minval=-bound, maxval=bound
                    # )
                    # self.weights[(in_k, in_p)][(out_k, out_p)] = rand_weights

                else:
                    # Works really well, not sure why?
                    filter_spatial_dims, _ = geom.parse_shape(
                        self.invariant_filters[filter_key].shape[1:], self.D
                    )
                    bound_shape = (in_c,) + filter_spatial_dims + (self.D,) * in_k
                    bound = 1 / jnp.sqrt(math.prod(bound_shape))
                    self.weights[(in_k, in_p)][(out_k, out_p)] = random.uniform(
                        subkey1,
                        shape=(out_c, in_c, len(self.invariant_filters[filter_key])),
                        minval=-bound,
                        maxval=bound,
                    )
                    all_filter_spatial_dims.append(filter_spatial_dims)

                if use_bias:
                    # this may get set multiple times, bound could be different but not a huge issue?
                    self.bias[(out_k, out_p)] = random.uniform(
                        subkey2,
                        shape=(out_c,) + (1,) * (self.D + out_k),
                        minval=-bound,
                        maxval=bound,
                    )

        # If all the in_c match, all out_c match, and all the filter dims match, can use fast_mode
        self.fast_mode = (
            (not self.missing_filter)
            and (len(set([in_c for _, in_c in input_keys])) == 1)
            and (len(set([out_c for _, out_c in target_keys])) == 1)
            and (len(set(all_filter_spatial_dims)) == 1)
        )
        self.fast_mode = False

    def fast_convolve(
        self: Self,
        input_layer: geom.Layer,
        weights: dict[LayerKey, dict[LayerKey, jax.Array]],
    ):
        """
        Convolve when all filter_spatial_dims, in_c, and out_c match, can do a single convolve
        instead of multiple between each type. Sadly, only ~20% speedup.
        """
        # These must all be equal to call fast_convolve
        in_c = self.input_keys[0][1]
        out_c = self.target_keys[0][1]

        one_img = next(iter(input_layer.values()))
        spatial_dims, _ = geom.parse_shape(one_img.shape[2:], self.D)
        batch = len(one_img)
        one_filter = next(iter(self.invariant_filters.values()))
        filter_spatial_dims, _ = geom.parse_shape(one_filter.shape[1:], self.D)

        image_ravel = jnp.zeros((batch,) + spatial_dims + (0, in_c))
        filter_ravel = jnp.zeros((in_c,) + filter_spatial_dims + (0, out_c))
        for (in_k, in_p), image_block in input_layer.items():
            # (batch,in_c,spatial,tensor) -> (batch,spatial,-1,in_c)
            img = jnp.moveaxis(image_block.reshape((batch, in_c) + spatial_dims + (-1,)), 1, -1)
            image_ravel = jnp.concatenate([image_ravel, img], axis=-2)

            filter_ravel_in = jnp.zeros(
                (in_c,) + filter_spatial_dims + (self.D,) * in_k + (0, out_c)
            )
            for (out_k, out_p), weight_block in weights[(in_k, in_p)].items():
                filter_key = (in_k + out_k, (in_p + out_p) % 2)

                # (out_c,in_c,num_filters),(num, spatial, tensor) -> (out_c,in_c,spatial,tensor)
                filter_block = jnp.einsum(
                    "ijk,k...->ij...",
                    weight_block,
                    jax.lax.stop_gradient(self.invariant_filters[filter_key]),
                )
                # (out_c,in_c,spatial,tensor) -> (in_c,spatial,in_tensor,-1,out_c)
                ff = jnp.moveaxis(
                    filter_block.reshape(
                        (out_c, in_c) + filter_spatial_dims + (self.D,) * in_k + (-1,)
                    ),
                    0,
                    -1,
                )
                filter_ravel_in = jnp.concatenate([filter_ravel_in, ff], axis=-2)

            filter_ravel_in = filter_ravel_in.reshape(
                (in_c,) + filter_spatial_dims + (-1,) + (out_c,)
            )
            filter_ravel = jnp.concatenate([filter_ravel, filter_ravel_in], axis=-2)

        image_ravel = image_ravel.reshape((batch,) + spatial_dims + (-1,))
        filter_ravel = jnp.moveaxis(filter_ravel, 0, self.D).reshape(
            filter_spatial_dims + (in_c, -1)
        )

        out = geom.convolve_ravel(
            self.D,
            image_ravel,
            filter_ravel,
            input_layer.is_torus,
            self.stride,
            self.padding,
            self.lhs_dilation,
            self.rhs_dilation,
        )
        new_spatial_dims = out.shape[1 : 1 + self.D]
        # (batch,spatial,tensor_sum*out_c) -> (batch,out_c,spatial,tensor_sum)
        out = jnp.moveaxis(out.reshape((batch,) + new_spatial_dims + (-1, out_c)), -1, 1)

        out_k_sum = sum([self.D**out_k for (out_k, _), _ in self.target_keys])
        idx = 0
        layer = input_layer.empty()
        for in_k, in_p in input_layer.keys():
            length = (self.D**in_k) * out_k_sum
            # break off all the channels related to this particular in_k
            out_per_in = out[..., idx : idx + length].reshape(
                (batch, out_c) + new_spatial_dims + (self.D,) * in_k + (-1,)
            )

            out_idx = 0
            for (out_k, out_p), _ in self.target_keys:
                out_length = self.D**out_k
                # separate the different out_k parts for particular in_k
                img_block = out_per_in[..., out_idx : out_idx + out_length]
                img_block = img_block.reshape(
                    (batch, out_c) + new_spatial_dims + (self.D,) * (in_k + out_k)
                )
                contracted_img = jnp.sum(img_block, axis=range(2 + self.D, 2 + self.D + in_k))

                if (out_k, out_p) in layer:  # it already has that key
                    layer[(out_k, out_p)] = contracted_img + layer[(out_k, out_p)]
                else:
                    layer.append(out_k, out_p, contracted_img)

                out_idx += out_length

            idx += length

        return layer

    def individual_convolve(
        self: Self,
        input_layer: geom.Layer,
        weights: dict[LayerKey, dict[LayerKey, jax.Array]],
    ):
        """
        Function to perform convolve_contract on an entire layer by doing the pairwise convolutions
        individually. This is necessary when filters have unequal sizes, or the in_c or out_c are
        not all equal. Weights is passed as an argument to make it easier to test this function.
        """
        layer = input_layer.empty()
        for (in_k, in_p), images_block in input_layer.items():
            for (out_k, out_p), weight_block in weights[(in_k, in_p)].items():
                filter_key = (in_k + out_k, (in_p + out_p) % 2)

                # (out_c,in_c,num_inv_filters) (num, spatial, tensor) -> (out_c,in_c,spatial,tensor)
                filter_block = jnp.einsum(
                    "ijk,k...->ij...",
                    weight_block,
                    jax.lax.stop_gradient(self.invariant_filters[filter_key]),
                )

                convolve_contracted_imgs = geom.convolve_contract(
                    input_layer.D,
                    images_block,  # add batch dim
                    filter_block,
                    input_layer.is_torus,
                    self.stride,
                    self.padding,
                    self.lhs_dilation,
                    self.rhs_dilation,
                )

                if (out_k, out_p) in layer:  # it already has that key
                    layer[(out_k, out_p)] = convolve_contracted_imgs + layer[(out_k, out_p)]
                else:
                    layer.append(out_k, out_p, convolve_contracted_imgs)

        return layer

    def __call__(self: Self, input_layer: geom.Layer):
        if self.fast_mode:
            layer = self.fast_convolve(input_layer, self.weights)
        else:  # slow mode
            layer = self.individual_convolve(input_layer, self.weights)

        if self.use_bias:
            biased_layer = layer.empty()
            for (k, p), image in layer.items():
                if (k, p) == (0, 0) and (self.use_bias == "scalar" or self.use_bias == "auto"):
                    biased_layer.append(k, p, image + self.bias[(k, p)])
                elif ((k, p) != (0, 0) and self.use_bias == "auto") or self.use_bias == "mean":
                    mean_image = jnp.mean(
                        image, axis=tuple(range(2, 2 + self.invariant_filters.D)), keepdims=True
                    )
                    biased_layer.append(
                        k,
                        p,
                        image + mean_image * self.bias[(k, p)],
                    )

            return biased_layer
        else:
            return layer


class GroupNorm(eqx.Module):
    scale: dict[LayerKey, jax.Array]
    bias: dict[LayerKey, jax.Array]
    vanilla_norm: dict[LayerKey, eqx.nn.GroupNorm]

    D: int = eqx.field(static=False)
    groups: int = eqx.field(static=False)
    eps: float = eqx.field(static=False)

    def __init__(
        self: Self,
        input_keys: tuple[tuple[LayerKey, int]],
        D: int,
        groups: int,
        eps: float = 1e-5,
    ) -> Self:
        """
        Implementation of group_norm. When num_groups=num_channels, this is equivalent to instance_norm. When
        num_groups=1, this is equivalent to layer_norm. This function takes in a BatchLayer, not a Layer.
        args:
            input_keys: input key signature
            D (int): dimension
            groups (int): the number of channel groups for group_norm
            eps (float): number to add to variance so we aren't dividing by 0
            equivariant (bool): defaults to True
        """
        self.D = D
        self.groups = groups
        self.eps = eps

        self.scale = {}
        self.bias = {}
        self.vanilla_norm = {}  # for scalars, can use basic implementation of GroupNorm
        for (k, p), in_c in input_keys:
            assert (
                in_c % groups
            ) == 0, f"group_norm: Groups must evenly divide channels, but got groups={groups}, channels={in_c}."

            if k == 0:
                self.vanilla_norm[(k, p)] = eqx.nn.GroupNorm(groups, in_c, eps)
            elif k == 1:
                self.scale[(k, p)] = jnp.ones((in_c,) + (1,) * (D + k))
                self.bias[(k, p)] = jnp.zeros((in_c,) + (1,) * (D + k))
            elif k > 1:
                raise NotImplementedError(
                    f"ml::group_norm: Equivariant group_norm not implemented for k>1, but k={k}",
                )

    def __call__(self: Self, x: geom.Layer) -> geom.Layer:
        out_x = x.empty()
        for (k, p), image_block in x.items():
            if k == 0:
                whitened_data = jax.vmap(self.vanilla_norm[(k, p)])(image_block)  # normal norm
            elif k == 1:
                # save mean vec, allows for un-mean centering (?)
                mean_vec = jnp.mean(image_block, axis=tuple(range(2, 2 + self.D)), keepdims=True)
                assert mean_vec.shape == image_block.shape[:2] + (1,) * self.D + (self.D,) * k
                whitened_data = _group_norm_K1(self.D, image_block, self.groups, eps=self.eps)
                whitened_data = whitened_data * self.scale[(k, p)] + self.bias[(k, p)] * mean_vec
            elif k > 1:
                raise NotImplementedError(
                    f"ml::group_norm: Equivariant group_norm not implemented for k>1, but k={k}",
                )

            out_x.append(k, p, whitened_data)

        return out_x


class LayerNorm(GroupNorm):

    def __init__(
        self: Self, input_keys: tuple[tuple[LayerKey, int]], D: int, eps: float = 1e-5
    ) -> Self:
        super(LayerNorm, self).__init__(input_keys, D, 1, eps)


class VectorNeuronNonlinear(eqx.Module):
    weights: dict[LayerKey, jax.Array]

    eps: float = eqx.field(static=True)
    D: int = eqx.field(static=True)
    scalar_activation: Callable = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: tuple[tuple[geom.LayerKey, int]],
        D: int,
        scalar_activation: Callable[[ArrayLike], jax.Array] = jax.nn.relu,
        eps: float = 1e-5,
        key: ArrayLike = None,
    ) -> Self:
        """
        The vector nonlinearity in the Vector Neurons paper: https://arxiv.org/pdf/2104.12229.pdf
        Basically use the channels of a vector to get a direction vector. Use the direction vector
        to get an inner product with the input vector. The inner product is like the input to a
        typical nonlinear activation, and it is used to scale the non-orthogonal part of the input
        vector.
        args:
            input_keys: the input keys to this layer
            scalar_activation (func): nonlinearity used for scalar
            eps (float): small value to avoid dividing by zero if the k_vec is close to 0, defaults to 1e-5
        """
        self.eps = eps
        self.D = D
        self.scalar_activation = scalar_activation

        self.weights = {}
        for (k, p), in_c in input_keys:
            if (k, p) != (0, 0):  # initialization?
                bound = 1.0 / jnp.sqrt(in_c)
                key, subkey = random.split(key, num=2)
                self.weights[(k, p)] = random.uniform(
                    subkey, shape=(in_c, in_c), minval=-bound, maxval=bound
                )

    def __call__(self: Self, x: geom.BatchLayer):
        out_x = x.empty()
        for (k, p), img_block in x.items():

            if (k, p) == (0, 0):
                out_x.append(k, p, self.scalar_activation(img_block))
            else:
                # -> (out_c,spatial,tensor)
                k_vec = jnp.einsum("ij,kj...->ki...", self.weights[(k, p)], img_block)
                k_vec_normed = k_vec / (geom.norm(2 + self.D, k_vec, keepdims=True) + self.eps)

                inner_prod = jnp.einsum(
                    f"...{geom.LETTERS[:k]},...{geom.LETTERS[:k]}->...", img_block, k_vec_normed
                )

                # split the vector into a parallel section and a perpendicular section
                v_parallel = jnp.einsum(
                    f"...,...{geom.LETTERS[:k]}->...{geom.LETTERS[:k]}", inner_prod, k_vec_normed
                )
                v_perp = img_block - v_parallel
                h = self.scalar_activation(inner_prod) / (jnp.abs(inner_prod) + self.eps)

                scaled_parallel = jnp.einsum(
                    f"...,...{geom.LETTERS[:k]}->...{geom.LETTERS[:k]}", h, v_parallel
                )
                out_x.append(k, p, scaled_parallel + v_perp)

        return out_x


class MaxNormPool(eqx.Module):
    patch_len: int = eqx.field(static=True)
    use_norm: bool = eqx.field(static=True)

    def __init__(self: Self, patch_len: int, use_norm: bool = True):
        self.patch_len = patch_len
        self.use_norm = use_norm

    def __call__(self: Self, x: geom.Layer):
        in_axes = (None, 0, None, None)
        vmap_max_pool = jax.vmap(jax.vmap(geom.max_pool, in_axes=in_axes), in_axes=in_axes)

        out_x = x.empty()
        for (k, p), image_block in x.items():
            out_x.append(k, p, vmap_max_pool(x.D, image_block, self.patch_len, self.use_norm))

        return out_x


class LayerWrapper(eqx.Module):
    modules: dict[LayerKey, Union[eqx.Module, Callable]]

    def __init__(
        self: Self, module: Union[eqx.Module, Callable], input_keys: tuple[tuple[LayerKey, int]]
    ):
        """
        Perform the module or callable (e.g., activation) on each layer of the input layer. Since
        we only take input_keys, module should preserve the shape/tensor order and parity.
        """
        self.modules = {}
        for (k, p), _ in input_keys:
            # I believe this *should* duplicate so they are independent, per the description in
            # https://docs.kidger.site/equinox/api/nn/shared/. However, it may not. In the scalar
            # case this should be perfectly fine though.
            self.modules[(k, p)] = module

    def __call__(self: Self, x: geom.Layer):
        out_layer = x.__class__({}, x.D, x.is_torus)
        for (k, p), image in x.items():
            vmap_call = eqx.filter_vmap(self.modules[(k, p)], axis_name="batch")
            out_layer.append(k, p, vmap_call(image))

        return out_layer


class LayerWrapperAux(eqx.Module):
    modules: dict[LayerKey, Union[eqx.Module, Callable]]

    def __init__(
        self: Self,
        module: Union[eqx.Module, Callable],
        input_keys: tuple[tuple[LayerKey, int]],
    ):
        """
        Perform the module or callable (e.g., activation) on each layer of the input layer. Since
        we only take input_keys, module should preserve the shape/tensor order and parity.
        """
        self.modules = {}
        for (k, p), _ in input_keys:
            # I believe this *should* duplicate so they are independent, per the description in
            # https://docs.kidger.site/equinox/api/nn/shared/. However, it may not. In the scalar
            # case this should be perfectly fine though.
            self.modules[(k, p)] = module

    def __call__(self: Self, x: geom.Layer, aux_data: Optional[eqx.nn.State]):
        out_layer = x.__class__({}, x.D, x.is_torus)
        for (k, p), image in x.items():
            vmap_call = eqx.filter_vmap(
                self.modules[(k, p)], in_axes=(0, None), out_axes=(0, None), axis_name="batch"
            )
            out, aux_data = vmap_call(image, aux_data)
            out_layer.append(k, p, out)

        return out_layer, aux_data


def save(filename, model):
    # TODO: save batch stats
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def _group_norm_K1(
    D: int, image_block: ArrayLike, groups: int, method: str = "eigh", eps: float = 1e-5
) -> jax.Array:
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
    batch, in_c = image_block.shape[:2]
    spatial_dims, k = geom.parse_shape(image_block.shape[2:], D)
    assert (
        k == 1
    ), f"ml::_group_norm_K1: Equivariant group_norm is not implemented for k>1, but k={k}"
    assert (in_c % groups) == 0  # groups must evenly divide the number of channels
    channels_per_group = in_c // groups

    image_grouped = image_block.reshape((batch, groups, channels_per_group) + spatial_dims + (D,))

    mean = jnp.mean(image_grouped, axis=tuple(range(2, 3 + D)), keepdims=True)  # (B,G,1,(1,)*D,D)
    centered_img = image_grouped - mean  # (B,G,in_c//G,spatial,tensor)

    X = centered_img.reshape((batch, groups, -1, D))  # (B,G,spatial*in_c//G,D)
    cov = jnp.einsum("...ij,...ik->...jk", X, X) / X.shape[-2]  # biased cov, (B,G,D,D)

    if method == "eigh":
        # symmetrize_input=True seems to cause issues with autograd, and cov is already symmetric
        eigvals, eigvecs = jnp.linalg.eigh(cov, symmetrize_input=False)
        eigvals_invhalf = jnp.sqrt(1.0 / (eigvals + eps))
        S_diag = jax.vmap(lambda S: jnp.diag(S))(eigvals_invhalf.reshape((-1, D))).reshape(
            (batch, groups, D, D)
        )
        # do U S U^T, and multiply each vector in centered_img by the resulting matrix
        whitened_data = jnp.einsum(
            "...ij,...jk,...kl,...ml->...mi",
            eigvecs,
            S_diag,
            eigvecs.transpose((0, 1, 3, 2)),
            centered_img.reshape((batch, groups, -1, D)),
        )
    elif method == "cholesky":
        L = jax.lax.linalg.cholesky(cov, symmetrize_input=False)  # (batch*groups,D,D)
        L = L + eps * jnp.eye(D).reshape((1, D, D))
        whitened_data = jax.lax.linalg.triangular_solve(
            L,
            centered_img.reshape((batch * groups, -1) + (D,)),
            left_side=False,
            lower=True,
        )
    else:
        raise NotImplementedError(f"ml::_group_norm_K1: method {method} not implemented.")

    return whitened_data.reshape(image_block.shape)


@functools.partial(jax.jit, static_argnums=1)
def average_pool_layer(input_layer: geom.Layer, patch_len: int) -> geom.Layer:
    out_layer = input_layer.empty()
    vmap_avg_pool = jax.vmap(geom.average_pool, in_axes=(None, 0, None))
    for (k, parity), image_block in input_layer.items():
        out_layer.append(k, parity, vmap_avg_pool(input_layer.D, image_block, patch_len))

    return out_layer


def batch_average_pool(input_layer: geom.BatchLayer, patch_len: int) -> geom.BatchLayer:
    return jax.vmap(average_pool_layer, in_axes=(0, None))(input_layer, patch_len)


## Losses


def timestep_smse_loss(
    layer_x: geom.BatchLayer,
    layer_y: geom.BatchLayer,
    n_steps: int,
    reduce: Optional[str] = "mean",
) -> jax.Array:
    """
    Returns loss for each timestep. Loss is summed over the channels, and mean over spatial dimensions
    and the batch.
    args:
        layer_x (BatchLayer): predicted data
        layer_y (BatchLayer): target data
        n_steps (int): number of timesteps, all channels should be a multiple of this
        reduce (str): how to reduce over the batch, one of mean or max, defaults to mean
    """
    assert reduce in {"mean", "max", None}
    spatial_size = np.multiply.reduce(layer_x.get_spatial_dims())
    batch = layer_x.get_L()
    loss_per_step = jnp.zeros((batch, n_steps))
    for image_a, image_b in zip(layer_x.values(), layer_y.values()):  # loop over image types
        image_a = image_a.reshape((batch, -1, n_steps) + image_a.shape[2:])
        image_b = image_b.reshape((batch, -1, n_steps) + image_b.shape[2:])
        loss = (
            jnp.sum((image_a - image_b) ** 2, axis=(1,) + tuple(range(3, image_a.ndim)))
            / spatial_size
        )
        loss_per_step = loss_per_step + loss

    if reduce == "mean":
        return jnp.mean(loss_per_step, axis=0)
    elif reduce == "max":
        return loss_per_step[jnp.argmax(jnp.sum(loss_per_step, axis=1))]
    elif reduce is None:
        return loss_per_step


def smse_loss(layer_x: geom.Layer, layer_y: geom.Layer) -> jax.Array:
    """
    Sum of mean squared error loss. The sum is over the channels, the mean is over the spatial dimensions and
    the batch.
    args:
        layer_x (Layer): the input layer or batch layer
        layer_y (Layer): the target layer or batch layer
    """
    spatial_size = np.multiply.reduce(layer_x.get_spatial_dims())
    return jnp.mean(
        jnp.sum((layer_x.to_vector() - layer_y.to_vector()) ** 2 / spatial_size, axis=1)
    )


def normalized_smse_loss(
    layer_x: geom.BatchLayer, layer_y: geom.BatchLayer, eps: float = 1e-5
) -> jax.Array:
    """
    Pointwise normalized loss. We find the norm of each channel at each spatial point of the true value
    and divide the tensor by that norm. Then we take the l2 loss, mean over the spatial dimensions, sum
    over the channels, then mean over the batch.
    args:
        layer_x (BatchLayer): input batch layer
        layer_y (BatchLayer): target batch layer
        eps (float): ensure that we aren't dividing by 0 norm, defaults to 1e-5
    """
    spatial_size = np.multiply.reduce(layer_x.get_spatial_dims())

    order_loss = jnp.zeros(layer_x.get_L())
    for (k, parity), img_block in layer_y.items():
        norm = geom.norm(layer_y.D + 2, img_block, keepdims=True) ** 2  # (b,c,spatial, (1,)*k)
        normalized_l2 = ((layer_x[(k, parity)] - img_block) ** 2) / (norm + eps)
        order_loss = order_loss + (
            jnp.sum(normalized_l2, axis=range(1, img_block.ndim)) / spatial_size
        )  # (b,)

    return jnp.mean(order_loss)


## Data and Batching operations


def get_batch_layer(
    layers: Union[Sequence[geom.BatchLayer], geom.BatchLayer],
    batch_size: int,
    rand_key: ArrayLike,
    devices: Optional[list[jax.Device]] = None,
) -> Union[list[list[geom.BatchLayer]], list[geom.BatchLayer]]:
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
    for i in range(int(math.floor(L / batch_size))):  # iterate through the batches of an epoch
        idxs = batch_indices[i * batch_size : (i + 1) * batch_size]
        for j, layer in enumerate(layers):
            batches[j].append(layer.get_subset(idxs).reshape_pmap(devices))

    return batches if (len(batches) > 1) else batches[0]


### Stopping Conditions


class StopCondition:
    def __init__(self: Self, verbose: int = 0) -> Self:
        assert verbose in {0, 1, 2}
        self.best_model = None
        self.verbose = verbose

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: int,
    ) -> None:
        pass

    def log_status(
        self: Self, epoch: int, train_loss: float, val_loss: float, epoch_time: int
    ) -> None:
        if train_loss is not None:
            if val_loss is not None:
                print(
                    f"Epoch {epoch} Train: {train_loss:.7f} Val: {val_loss:.7f} Epoch time: {epoch_time:.5f}",
                )
            else:
                print(f"Epoch {epoch} Train: {train_loss:.7f} Epoch time: {epoch_time:.5f}")


class EpochStop(StopCondition):
    # Stop when enough epochs have passed.

    def __init__(self: Self, epochs: int, verbose: int = 0) -> Self:
        super(EpochStop, self).__init__(verbose=verbose)
        self.epochs = epochs

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: float,
        val_loss: float,
        epoch_time: int,
    ) -> bool:
        self.best_model = model

        if self.verbose == 2 or (
            self.verbose == 1 and (current_epoch % (self.epochs // np.min([10, self.epochs])) == 0)
        ):
            self.log_status(current_epoch, train_loss, val_loss, epoch_time)

        return current_epoch >= self.epochs


class TrainLoss(StopCondition):
    # Stop when the training error stops improving after patience number of epochs.

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> Self:
        super(TrainLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_train_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        epoch_time: int,
    ) -> bool:
        if train_loss is None:
            return False

        if train_loss < (self.best_train_loss - self.min_delta):
            self.best_train_loss = train_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, epoch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience


class ValLoss(StopCondition):
    # Stop when the validation error stops improving after patience number of epochs.

    def __init__(self: Self, patience: int = 0, min_delta: float = 0, verbose: int = 0) -> Self:
        super(ValLoss, self).__init__(verbose=verbose)
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_loss = jnp.inf
        self.epochs_since_best = 0

    def stop(
        self: Self,
        model: eqx.Module,
        current_epoch: int,
        train_loss: Optional[float],
        val_loss: Optional[float],
        epoch_time: int,
    ) -> bool:
        if val_loss is None:
            return False

        if val_loss < (self.best_val_loss - self.min_delta):
            self.best_val_loss = val_loss
            self.best_model = model
            self.epochs_since_best = 0

            if self.verbose >= 1:
                self.log_status(current_epoch, train_loss, val_loss, epoch_time)
        else:
            self.epochs_since_best += 1

        return self.epochs_since_best > self.patience


# ~~~~~~~~~~~~~~~~~~~~~~ Training Functions ~~~~~~~~~~~~~~~~~~~~~~


def autoregressive_step(
    input: geom.BatchLayer,
    one_step: geom.BatchLayer,
    output: geom.BatchLayer,
    past_steps: int,
    constant_fields: dict[LayerKey, ArrayLike] = {},
    future_steps: int = 1,
) -> tuple[geom.BatchLayer, geom.BatchLayer]:
    """
    Given the input layer, the next step of the model, and the output, update the input
    and output to be fed into the model next. Batch Layers should have shape (batch,channels,spatial,tensor).
    Channels are c*past_steps + constant_steps where c is some positive integer.
    args:
        input (BatchLayer): the input to the model
        one_step (BatchLayer): the model output at this step, assumed to be a single time step
        output (BatchLayer): the full output that we are building up
        past_steps (int): the number of past time steps that are fed into the model
        constant_fields (dict): a map {key:n_constant_fields} for fields that don't depend on timestep
        future_steps (int): number of future steps that the model outputs, currently must be 1
    """
    assert (
        future_steps == 1
    ), f"ml::autoregressive_step: future_steps must be 1, but found {future_steps}."

    new_input = input.empty()
    new_output = output.empty()
    for key, step_data in one_step.items():
        k, parity = key
        batch = step_data.shape[0]
        img_shape = step_data.shape[2:]  # the shape of the image, spatial + tensor
        exp_data = step_data.reshape((batch, -1, future_steps) + img_shape)
        n_channels = exp_data.shape[1]  # number of channels for the key, not timesteps

        if (key in constant_fields) and constant_fields[key]:
            n_const_fields = constant_fields[key]
            const_fields = input[key][:, -n_const_fields:]
        else:
            n_const_fields = 0
            const_fields = jnp.zeros((batch, 0) + img_shape)

        exp_input = input[key][:, : (-n_const_fields or None)].reshape(
            (batch, -1, past_steps) + img_shape
        )
        next_input = jnp.concatenate([exp_input[:, :, 1:], exp_data], axis=2).reshape(
            (batch, -1) + img_shape
        )
        new_input.append(k, parity, jnp.concatenate([next_input, const_fields], axis=1))

        if key in output:
            exp_output = output[key].reshape((batch, n_channels, -1) + img_shape)
            full_output = jnp.concatenate([exp_output, exp_data], axis=2).reshape(
                (batch, -1) + img_shape
            )
        else:
            full_output = step_data

        new_output.append(k, parity, full_output)

    return new_input, new_output


def autoregressive_map(
    batch_model: eqx.Module,
    x: geom.BatchLayer,
    aux_data: Any = None,
    past_steps: int = 1,
    future_steps: int = 1,
) -> geom.BatchLayer:
    """
    Given a model, perform an autoregressive step (future_steps) times, and return the output
    steps in a single layer.
    args:
        batch_model (eqx.Module): model that operates of batches, probably a vmapped version of model.
        x (BatchLayer): the input layer to map
        past_steps (int): the number of past steps input to the autoregressive map, default 1
        future_steps (int): how many times to loop through the autoregression, default 1
        aux_data (): auxilliary data to pass to the network
        has_aux (bool): whether net returns an aux_data, defaults to False
    """
    out_x = x.empty()  # assume out_layer matches D and is_torus
    for _ in range(future_steps):
        if aux_data is None:
            learned_x = batch_model(x)
        else:
            learned_x, aux_data = batch_model(x, aux_data)

        x, out_x = autoregressive_step(x, learned_x, out_x, past_steps)

    return out_x, aux_data


def evaluate(
    model: eqx.Module,
    map_and_loss: Union[
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, eqx.nn.State],
            tuple[jax.Array, eqx.nn.State, geom.BatchLayer],
        ],
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, eqx.nn.State],
            tuple[jax.Array, eqx.nn.State],
        ],
    ],
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> jax.Array:
    """
    Runs map_and_loss for the entire layer_X, layer_Y, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        model (model PyTree): the model to run through map_and_loss
        x (BatchLayer): input data
        y (BatchLayer): target output data
        sharding: sharding over multiple GPUs, if None (default), will use available devices
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
    returns: average loss over the entire layer
    """
    inference_model = eqx.nn.inference_mode(model)
    if return_map:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None, 0),
        )
        loss, _, out_layer = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0), out_layer.merge_pmap()
    else:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None),
        )
        loss, _ = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0)


def loss_reducer(ls):
    """
    A reducer for map_loss_in_batches that takes the batch mean of the loss
    """
    return jnp.mean(jnp.stack(ls), axis=0)


def aux_data_reducer(ls):
    """
    A reducer for aux_data like batch stats that just takes the last one
    """
    return ls[-1]


def layer_reducer(ls):
    """
    If map data returns the mapped layers, merge them togther
    """
    return functools.reduce(lambda carry, val: carry.concat(val), ls, ls[0].empty())


def map_loss_in_batches(
    map_and_loss: Callable[
        [eqx.Module, geom.BatchLayer, geom.BatchLayer, eqx.nn.State], tuple[jax.Array, eqx.nn.State]
    ],
    model: eqx.Module,
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    batch_size: int,
    rand_key: ArrayLike,
    reducers: Optional[tuple] = None,
    devices: Optional[list[jax.devices]] = None,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> jax.Array:
    """
    Runs map_and_loss for the entire layer_X, layer_Y, splitting into batches if the layer is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model (model PyTree): the model to run through map_and_loss
        x (BatchLayer): input data
        y (BatchLayer): target output data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        devices (list of jax devices): the gpus that the code will run on
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
    returns: average loss over the entire layer
    """
    if reducers is None:
        # use the default reducer for loss
        reducers = [loss_reducer]
        if return_map:
            reducers.append(layer_reducer)

    X_batches, Y_batches = get_batch_layer((x, y), batch_size, rand_key, devices)
    results = [[] for _ in range(len(reducers))]
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        one_result = evaluate(model, map_and_loss, X_batch, Y_batch, aux_data, return_map)

        if len(reducers) == 1:
            results[0].append(one_result)
        else:
            for val, result_ls in zip(one_result, results):
                result_ls.append(val)

    if len(reducers) == 1:
        return reducers[0](results[0])
    else:
        return tuple(reducer(result_ls) for reducer, result_ls in zip(reducers, results))


def train_step(
    map_and_loss: Callable[
        [eqx.Module, geom.BatchLayer, geom.BatchLayer, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state,
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    aux_data: Optional[eqx.nn.State] = None,
):
    """
    Perform one step and gradient update of the model. Uses filter_pmap to use multiple gpus.
    args:
        map_and_loss (func): map and loss function where the input is a model pytree, x BatchLayer,
            y BatchLayer, and aux_data, and returns a float loss and aux_data
        model (equinox model pytree): the model
        optim (optax optimizer):
        opt_state:
        x (BatchLayer): input data
        y (BatchLayer): target data
        aux_data (Any): auxilliary data for stateful layers
    returns: model, opt_state, loss_value
    """
    # NOTE: do not `jit` over `pmap` see (https://github.com/google/jax/issues/2926)
    loss_grad = eqx.filter_value_and_grad(map_and_loss, has_aux=True)

    compute_loss_pmap = eqx.filter_pmap(
        loss_grad,
        axis_name="pmap_batch",
        in_axes=(None, 0, 0, None),
        out_axes=((0, None), 0),
    )
    (loss, aux_data), grads = compute_loss_pmap(model, x, y, aux_data)
    loss = jnp.mean(loss, axis=0)

    get_weights = lambda m: jax.tree_util.tree_leaves(m, is_leaf=eqx.is_array)
    new_grad_arrays = [jnp.mean(x, axis=0) for x in get_weights(grads)]
    grads = eqx.tree_at(get_weights, grads, new_grad_arrays)

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss, aux_data


def train(
    X: geom.BatchLayer,
    Y: geom.BatchLayer,
    map_and_loss: Union[
        Callable[[eqx.Module, geom.BatchLayer, geom.BatchLayer], jax.Array],
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, Any],
            tuple[jax.Array, Any],
        ],
    ],
    model: eqx.Module,
    rand_key: ArrayLike,
    stop_condition: StopCondition,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    validation_X: Optional[geom.BatchLayer] = None,
    validation_Y: Optional[geom.BatchLayer] = None,
    save_model: Optional[str] = None,
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> Union[tuple[eqx.Module, Any, jax.Array, jax.Array], tuple[eqx.Module, jax.Array, jax.Array]]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    shards over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    args:
        X (BatchLayer): The X input data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        Y (BatchLayer): The Y target data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss (function): function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        rand_key (jnp.random key): key for randomness
        stop_condition (StopCondition): when to stop the training process, currently only 1 condition
            at a time
        batch_size (int): the size of each mini-batch in SGD
        optimizer (optax optimizer): optimizer
        validation_X (BatchLayer): input data for a validation data set as a layer by k
            of (images, channels, (N,)*D, (D,)*k)
        validation_Y (BatchLayer): target data for a validation data set as a layer by k
            of (images, channels, (N,)*D, (D,)*k)
        save_model (str): if string, save model every 10 epochs, defaults to None
        aux_data (eqx.nn.State): initial aux data passed in to map_and_loss when has_aux is true.
        devices (list): gpu/cpu devices to use, if None (default) then it will use jax.devices()
    returns: A tuple of best model in inference mode, aux_data, epoch loss, and val loss
    """
    if isinstance(stop_condition, ValLoss) and not (validation_X and validation_Y):
        raise ValueError("Stop condition is ValLoss, but no validation data provided.")

    devices = devices if devices else jax.devices()

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    val_loss = 0
    epoch_time = 0
    while not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, epoch_time):
        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batch_layer((X, Y), batch_size, subkey, devices)
        epoch_loss = 0
        start_time = time.time()
        for X_batch, Y_batch in zip(X_batches, Y_batches):
            model, opt_state, loss_value, aux_data = train_step(
                map_and_loss,
                model,
                optimizer,
                opt_state,
                X_batch,
                Y_batch,
                aux_data,
            )
            epoch_loss += loss_value

        epoch_loss = epoch_loss / len(X_batches)
        epoch += 1

        # We evaluate the validation loss in batches for memory reasons.
        if validation_X and validation_Y:
            epoch_val_loss = map_loss_in_batches(
                map_and_loss,
                model,
                validation_X,
                validation_Y,
                batch_size,
                subkey,
                devices=devices,
                aux_data=aux_data,
            )
            val_loss = epoch_val_loss

        if save_model and ((epoch % 10) == 0):
            save(save_model, model)

        epoch_time = time.time() - start_time

    return stop_condition.best_model, aux_data, epoch_loss, val_loss


BENCHMARK_DATA = "benchmark_data"
BENCHMARK_MODEL = "benchmark_model"
BENCHMARK_NONE = "benchmark_none"


def benchmark(
    get_data: Callable[[Any], geom.BatchLayer],
    models: list[tuple[str, Callable[[geom.BatchLayer, ArrayLike, str], Any]]],
    rand_key: ArrayLike,
    benchmark: str,
    benchmark_range: Sequence,
    benchmark_type: str = BENCHMARK_DATA,
    num_trials: int = 1,
    num_results: int = 1,
) -> np.ndarray:
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
    assert benchmark_type in {BENCHMARK_DATA, BENCHMARK_MODEL, BENCHMARK_NONE}
    if benchmark_type == BENCHMARK_NONE:
        benchmark = ""
        benchmark_range = [0]

    results = np.zeros((num_trials, len(benchmark_range), len(models), num_results))
    for i in range(num_trials):
        for j, benchmark_val in enumerate(benchmark_range):

            data_kwargs = {benchmark: benchmark_val} if benchmark_type == BENCHMARK_DATA else {}
            model_kwargs = {benchmark: benchmark_val} if benchmark_type == BENCHMARK_MODEL else {}

            rand_key, subkey = random.split(rand_key)
            data = get_data(subkey, **data_kwargs)

            for k, (model_name, model) in enumerate(models):
                print(f"trial {i} {benchmark}: {benchmark_val} {model_name}")

                rand_key, subkey = random.split(rand_key)
                res = model(
                    data,
                    subkey,
                    f"{model_name}_{benchmark}{benchmark_val}_t{i}",
                    **model_kwargs,
                )

                if num_results > 1:
                    for q in range(num_results):
                        results[i, j, k, q] = res[q]
                else:
                    results[i, j, k, 0] = res

    return results
