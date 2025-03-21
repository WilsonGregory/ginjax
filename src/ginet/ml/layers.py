import math
import functools
from typing_extensions import Any, Callable, Optional, Self, Union

import jax
import jax.numpy as jnp
import jax.random as random
from jaxtyping import ArrayLike
import equinox as eqx

import ginet.geometric as geom


# ~~~~~~~~~~~~~~~~~~~~~~ Helpers ~~~~~~~~~~~~~~~~~~~~~~
def _group_norm_K1(
    D: int, image_block: jax.Array, groups: int, method: str = "eigh", eps: float = 1e-5
) -> jax.Array:
    """
    Perform the layer norm whitening on a vector image block. This is somewhat based on the Clifford
    Layers Batch norm, link below. However, this differs in that we use eigh rather than cholesky because
    cholesky is not invariant to all the elements of our group.
    https://github.com/microsoft/cliffordlayers/blob/main/cliffordlayers/nn/functional/batchnorm.py

    args:
        D: the dimension of the space
        image_block: data block of shape (channels,spatial,tensor)
        groups: the number of channel groups, must evenly divide channels
        method: method used for the whitening, either 'eigh', or 'cholesky'. Note that
            'cholesky' is not equivariant.
        eps: to avoid non-invertible matrices, added to the covariance matrix

    returns:
        the whitened data, shape (channels,spatial,tensor)
    """
    in_c = len(image_block)
    spatial_dims, k = geom.parse_shape(image_block.shape[1:], D)
    assert (
        k == 1
    ), f"ml::_group_norm_K1: Equivariant group_norm is not implemented for k>1, but k={k}"
    assert (in_c % groups) == 0  # groups must evenly divide the number of channels
    channels_per_group = in_c // groups

    image_grouped = image_block.reshape((groups, channels_per_group) + spatial_dims + (D,))

    mean = jnp.mean(image_grouped, axis=tuple(range(1, 2 + D)), keepdims=True)  # (G,1,(1,)*D,D)
    centered_img = image_grouped - mean  # (G,in_c//G,spatial,tensor)

    X = centered_img.reshape((groups, -1, D))  # (G,spatial*in_c//G,D)
    cov = jnp.einsum("...ij,...ik->...jk", X, X) / X.shape[-2]  # biased cov, (G,D,D)

    if method == "eigh":
        # symmetrize_input=True seems to cause issues with autograd, and cov is already symmetric
        eigvals, eigvecs = jnp.linalg.eigh(cov, symmetrize_input=False)  # (G,D), (G,D,D)
        eigvals_invhalf = jnp.sqrt(1.0 / (eigvals + eps))  # (G,D)
        S_diag = jax.vmap(lambda S: jnp.diag(S))(eigvals_invhalf).reshape((groups, D, D))
        # do U S U^T, and multiply each vector in centered_img by the resulting matrix
        whitened_data = jnp.einsum(
            "...ij,...jk,...kl,...ml->...mi",
            eigvecs,
            S_diag,
            eigvecs.transpose((0, 2, 1)),
            centered_img.reshape((groups, -1, D)),
        )
    elif method == "cholesky":
        L = jax.lax.linalg.cholesky(cov, symmetrize_input=False)  # (groups,D,D)
        L = L + eps * jnp.eye(D).reshape((1, D, D))
        whitened_data = jax.lax.linalg.triangular_solve(
            L,
            centered_img.reshape((groups, -1, D)),
            left_side=False,
            lower=True,
        )
    else:
        raise NotImplementedError(f"ml::_group_norm_K1: method {method} not implemented.")

    return whitened_data.reshape(image_block.shape)


# ~~~~~~~~~~~~~~~~~~~~~~ Layers ~~~~~~~~~~~~~~~~~~~~~~
class ConvContract(eqx.Module):
    """
    A layer then performs the convolution followed by contraction.
    """

    weights: dict[tuple[int, int], dict[tuple[int, int], jax.Array]]
    bias: dict[tuple[int, int], jax.Array]
    invariant_filters: geom.MultiImage

    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    use_bias: Union[str, bool] = eqx.field(static=True)
    stride: Union[int, tuple[int, ...]] = eqx.field(static=True)
    padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = eqx.field(static=True)
    lhs_dilation: Optional[tuple[int, ...]] = eqx.field(static=True)
    rhs_dilation: Union[int, tuple[int, ...]] = eqx.field(static=True)
    D: int = eqx.field(static=True)
    fast_mode: bool = eqx.field(static=True)
    missing_filter: bool = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        invariant_filters: geom.MultiImage,
        use_bias: Union[str, bool] = "auto",
        stride: Union[int, tuple[int, ...]] = 1,
        padding: Optional[Union[str, int, tuple[tuple[int, int], ...]]] = None,
        lhs_dilation: Optional[tuple[int, ...]] = None,
        rhs_dilation: Union[int, tuple[int, ...]] = 1,
        key: Any = None,
    ):
        """
        Constructor for equivariant tensor convolution then contraction.

        args:
            input_keys: A mapping of (k,p) to an integer representing the input channels
            target_keys: A mapping of (k,p) to an integer representing the output channels
            invariant_filters: A MultiImage of the invariant filters to build the convolution filters
            use_bias: One of 'auto', 'mean', or 'scalar', or True for 'auto' or False for no bias.
                Mean uses a mean scale for every type, scalar uses a regular bias for scalars only
                and auto does regular bias for scalars and mean for non-scalars.
            stride: convolution stride, defaults to (1,)*self.D
            padding: either 'TORUS','VALID', 'SAME', or D length tuple of (upper,lower) pairs,
                defaults to 'TORUS' if image.is_torus, else 'SAME'
            lhs_dilation: amount of dilation to apply to image in each dimension D, also transposed conv
            rhs_dilation: amount of dilation to apply to filter in each dimension D
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
        input_multi_image: geom.MultiImage,
        weights: dict[tuple[int, int], dict[tuple[int, int], jax.Array]],
    ) -> geom.MultiImage:
        """
        Convolve when all filter_spatial_dims, in_c, and out_c match, can do a single convolve
        instead of multiple between each type. Sadly, only ~20% speedup.
        """
        # These must all be equal to call fast_convolve
        in_c = self.input_keys[0][1]
        out_c = self.target_keys[0][1]

        one_img = next(iter(input_multi_image.values()))
        spatial_dims, _ = geom.parse_shape(one_img.shape[1:], self.D)
        one_filter = next(iter(self.invariant_filters.values()))
        filter_spatial_dims, _ = geom.parse_shape(one_filter.shape[1:], self.D)

        image_ravel = jnp.zeros(spatial_dims + (0, in_c))
        filter_ravel = jnp.zeros((in_c,) + filter_spatial_dims + (0, out_c))
        for (in_k, in_p), image_block in input_multi_image.items():
            # (in_c,spatial,tensor) -> (spatial,-1,in_c)
            img = jnp.moveaxis(image_block.reshape((in_c,) + spatial_dims + (-1,)), 0, -1)
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

        image_ravel = image_ravel.reshape(spatial_dims + (-1,))
        filter_ravel = jnp.moveaxis(filter_ravel, 0, self.D).reshape(
            filter_spatial_dims + (in_c, -1)
        )

        out = geom.convolve_ravel(
            self.D,
            image_ravel[None],  # add batch dim
            filter_ravel,
            input_multi_image.is_torus,
            self.stride,
            self.padding,
            self.lhs_dilation,
            self.rhs_dilation,
        )[0]
        new_spatial_dims = out.shape[: self.D]
        # (spatial,tensor_sum*out_c) -> (out_c,spatial,tensor_sum)
        out = jnp.moveaxis(out.reshape(new_spatial_dims + (-1, out_c)), -1, 0)

        out_k_sum = sum([self.D**out_k for (out_k, _), _ in self.target_keys])
        idx = 0
        out_multi_image = input_multi_image.empty()
        for in_k, in_p in input_multi_image.keys():
            length = (self.D**in_k) * out_k_sum
            # break off all the channels related to this particular in_k
            out_per_in = out[..., idx : idx + length].reshape(
                (out_c,) + new_spatial_dims + (self.D,) * in_k + (-1,)
            )

            out_idx = 0
            for (out_k, out_p), _ in self.target_keys:
                out_length = self.D**out_k
                # separate the different out_k parts for particular in_k
                img_block = out_per_in[..., out_idx : out_idx + out_length]
                img_block = img_block.reshape(
                    (out_c,) + new_spatial_dims + (self.D,) * (in_k + out_k)
                )
                contracted_img = jnp.sum(img_block, axis=range(1 + self.D, 1 + self.D + in_k))

                if (out_k, out_p) in out_multi_image:  # it already has that key
                    out_multi_image[(out_k, out_p)] = (
                        contracted_img + out_multi_image[(out_k, out_p)]
                    )
                else:
                    out_multi_image.append(out_k, out_p, contracted_img)

                out_idx += out_length

            idx += length

        return out_multi_image

    def individual_convolve(
        self: Self,
        input_multi_image: geom.MultiImage,
        weights: dict[tuple[int, int], dict[tuple[int, int], jax.Array]],
    ) -> geom.MultiImage:
        """
        Function to perform convolve_contract on an entire MultiImage by doing the pairwise convolutions
        individually. This is necessary when filters have unequal sizes, or the in_c or out_c are
        not all equal. Weights is passed as an argument to make it easier to test this function.

        args:
            input_multi_image: the input
            weights: the weights used to combine the invariant filters

        returns:
            the convolved MultiImage
        """
        out = input_multi_image.empty()
        for (in_k, in_p), images_block in input_multi_image.items():
            for (out_k, out_p), weight_block in weights[(in_k, in_p)].items():
                filter_key = (in_k + out_k, (in_p + out_p) % 2)

                # (out_c,in_c,num_inv_filters) (num, spatial, tensor) -> (out_c,in_c,spatial,tensor)
                filter_block = jnp.einsum(
                    "ijk,k...->ij...",
                    weight_block,
                    jax.lax.stop_gradient(self.invariant_filters[filter_key]),
                )

                convolve_contracted_imgs = geom.convolve_contract(
                    input_multi_image.D,
                    images_block[None],  # add batch dim
                    filter_block,
                    input_multi_image.is_torus,
                    self.stride,
                    self.padding,
                    self.lhs_dilation,
                    self.rhs_dilation,
                )[0]

                if (out_k, out_p) in out:  # it already has that key
                    out[(out_k, out_p)] = convolve_contracted_imgs + out[(out_k, out_p)]
                else:
                    out.append(out_k, out_p, convolve_contracted_imgs)

        return out

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        The callable, calls either fast_convolve or individual_convolve. Currently fast_convolve
        is not used because it is not much faster.

        args:
            x: the input

        returns:
            the convolved MultiImage, which is a new object
        """
        if self.fast_mode:
            x = self.fast_convolve(x, self.weights)
        else:  # slow mode
            x = self.individual_convolve(x, self.weights)

        if self.use_bias:
            biased_x = x.empty()
            for (k, p), image in x.items():
                if (k, p) == (0, 0) and (self.use_bias == "scalar" or self.use_bias == "auto"):
                    biased_x.append(k, p, image + self.bias[(k, p)])
                elif ((k, p) != (0, 0) and self.use_bias == "auto") or self.use_bias == "mean":
                    mean_image = jnp.mean(
                        image, axis=tuple(range(1, 1 + self.invariant_filters.D)), keepdims=True
                    )
                    biased_x.append(k, p, image + mean_image * self.bias[(k, p)])

            return biased_x
        else:
            return x


class GroupNorm(eqx.Module):
    """
    Implementation of GroupNorm for equivariant and non-equivariant models.
    """

    scale: dict[tuple[int, int], jax.Array]
    bias: dict[tuple[int, int], jax.Array]
    vanilla_norm: dict[tuple[int, int], eqx.nn.GroupNorm]

    D: int = eqx.field(static=False)
    groups: int = eqx.field(static=False)
    eps: float = eqx.field(static=False)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        D: int,
        groups: int,
        eps: float = 1e-5,
    ) -> None:
        """
        Constructor for GroupNorm. When num_groups=num_channels, this is equivalent to instance_norm. When
        num_groups=1, this is equivalent to layer_norm.

        args:
            input_keys: input key signature
            D: dimension
            groups: the number of channel groups for group_norm
            eps: number to add to variance so we aren't dividing by 0
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

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for GroupNorm,

        args:
            x: input MultiImage

        returns:
            the output normed MultiImage
        """
        out_x = x.empty()
        for (k, p), image_block in x.items():
            if k == 0:
                whitened_data = self.vanilla_norm[(k, p)](image_block)  # normal norm
            elif k == 1:
                # save mean vec, allows for un-mean centering (?)
                mean_vec = jnp.mean(image_block, axis=tuple(range(1, 1 + self.D)), keepdims=True)
                assert mean_vec.shape == (image_block.shape[0],) + (1,) * self.D + (self.D,) * k
                whitened_data = _group_norm_K1(self.D, image_block, self.groups, eps=self.eps)
                whitened_data = whitened_data * self.scale[(k, p)] + self.bias[(k, p)] * mean_vec
            else:  # k > 1
                raise NotImplementedError(
                    f"ml::group_norm: Equivariant group_norm not implemented for k>1, but k={k}",
                )

            out_x.append(k, p, whitened_data)

        return out_x


class LayerNorm(GroupNorm):
    """
    LayerNorm, which is GroupNorm with a single group.
    """

    def __init__(self: Self, input_keys: geom.Signature, D: int, eps: float = 1e-5) -> None:
        """
        Constructor for LayerNorm.

        args:
            input_keys: the input signature
            D: the dimension
            eps: number to add to variance so we aren't dividing by 0
        """
        super(LayerNorm, self).__init__(input_keys, D, 1, eps)


class VectorNeuronNonlinear(eqx.Module):
    """
    The vector nonlinearity in the Vector Neurons paper: https://arxiv.org/pdf/2104.12229.pdf
    Basically use the channels of a vector to get a direction vector. Use the direction vector
    to get an inner product with the input vector. The inner product is like the input to a
    typical nonlinear activation, and it is used to scale the non-orthogonal part of the input
    vector.
    """

    weights: dict[tuple[int, int], jax.Array]

    eps: float = eqx.field(static=True)
    D: int = eqx.field(static=True)
    scalar_activation: Callable = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        D: int,
        scalar_activation: Callable[[ArrayLike], jax.Array] = jax.nn.relu,
        eps: float = 1e-5,
        key: Any = None,
    ) -> None:
        """
        Constructor for VectorNeuronNonlinear.

        args:
            input_keys: the signature of the input MultiImage
            D: the dimension
            scalar_activation: nonlinearity used for scalars
            eps: small value to avoid dividing by zero if the k_vec is close to 0
            key: jax.random key
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

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for VectorNeuronNonlinearity

        args:
            x: the input

        returns:
            a new MultiImage output
        """
        out_x = x.empty()
        for (k, p), img_block in x.items():

            if (k, p) == (0, 0):
                out_x.append(k, p, self.scalar_activation(img_block))
            else:
                # -> (out_c,spatial,tensor)
                k_vec = jnp.einsum("ij,j...->i...", self.weights[(k, p)], img_block)
                k_vec_normed = k_vec / (geom.norm(1 + self.D, k_vec, keepdims=True) + self.eps)

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
    """
    Layer that performs that MaxPool based on the norm of the tensor.
    """

    patch_len: int = eqx.field(static=True)
    use_norm: bool = eqx.field(static=True)

    def __init__(self: Self, patch_len: int, use_norm: bool = True) -> None:
        """
        Constructor for MaxNormPool.

        args:
            patch_len: sidelength of the patch
            use_norm: whether to use norm to calculate the max
        """
        self.patch_len = patch_len
        self.use_norm = use_norm

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for MaxNormPool.

        args:
            x: the input to the layer

        returns:
            a new max normed output MultiImage
        """
        in_axes = (None, 0, None, None)
        vmap_max_pool = jax.vmap(geom.max_pool, in_axes=in_axes)

        out_x = x.empty()
        for (k, p), image_block in x.items():
            out_x.append(k, p, vmap_max_pool(x.D, image_block, self.patch_len, self.use_norm))

        return out_x


class LayerWrapper(eqx.Module):
    """
    Wrapper class for any module which takes an image and converts it to taking and producing a
    MultiImage.
    """

    modules: dict[tuple[int, int], Callable[..., Any]]

    def __init__(self: Self, module: Callable[..., Any], input_keys: geom.Signature) -> None:
        """
        Perform the module or callable (e.g., activation) on each layer of the input MultiImage.
        Since we only take input_keys, module should preserve the shape/tensor order and parity.

        args:
            module: module should have as input/output an image of shape (channels, spatial)
            input_keys: actual input (and output) signature this module will process
        """
        self.modules = {}
        for (k, p), _ in input_keys:
            # I believe this *should* duplicate so they are independent, per the description in
            # https://docs.kidger.site/equinox/api/nn/shared/. However, it may not. In the scalar
            # case this should be perfectly fine though.
            self.modules[(k, p)] = module

    def __call__(self: Self, x: geom.MultiImage) -> geom.MultiImage:
        """
        Callable for LayerWrapper.

        args:
            x: the input

        returns:
            a new MultiImage
        """
        out = x.__class__({}, x.D, x.is_torus)
        for (k, p), image in x.items():
            out.append(k, p, self.modules[(k, p)](image))

        return out


class LayerWrapperAux(eqx.Module):
    """
    Wrapper class for any module which takes an image and aux data and converts it to taking and
    producing a MultiImage and aux data.
    """

    modules: dict[tuple[int, int], Callable[..., Any]]

    def __init__(self: Self, module: Callable[..., Any], input_keys: geom.Signature):
        """
        Perform the module or callable (e.g., activation) on each layer of the input MultiImage.
        Since we only take input_keys, module should preserve the shape/tensor order and parity.

        args:
            module: module should have as input/output an image of shape (channels, spatial) and
                aux data (likely batch_stats for BatchNorm).
            input_keys: actual input (and output) signature this module will process
        """
        self.modules = {}
        for (k, p), _ in input_keys:
            # I believe this *should* duplicate so they are independent, per the description in
            # https://docs.kidger.site/equinox/api/nn/shared/. However, it may not. In the scalar
            # case this should be perfectly fine though.
            self.modules[(k, p)] = module

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: Optional[eqx.nn.State]
    ) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
        """
        Callable for LayerWrapperAux.

        args:
            x: the input
            aux_data: the aux_data, e.g. for BatchNorm

        returns:
            a new MultiImage and the aux_data
        """
        out = x.__class__({}, x.D, x.is_torus)
        for (k, p), image in x.items():
            out_image, aux_data = self.modules[(k, p)](image, aux_data)
            out.append(k, p, out_image)

        return out, aux_data
