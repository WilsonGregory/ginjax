import time
import math
import functools
from typing import Any, Callable, Optional, Sequence, Union
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
import jax.experimental.mesh_utils as mesh_utils
from jax.typing import ArrayLike
import equinox as eqx
import optax

import geometricconvolutions.geometric as geom
import geometricconvolutions.ml as ml


# ~~~~~~~~~~~~~~~~~~~~~~ Layers ~~~~~~~~~~~~~~~~~~~~~~
class ConvContract(eqx.Module):
    weights: dict[ml.LayerKey, dict[ml.LayerKey, jax.Array]]
    bias: dict[ml.LayerKey, jax.Array]

    input_keys: tuple[tuple[ml.LayerKey, int]] = eqx.field(static=True)
    target_keys: tuple[tuple[ml.LayerKey, int]] = eqx.field(static=True)
    invariant_filters: geom.Layer = eqx.field(static=True)
    use_bias: Union[str, bool] = eqx.field(static=True)
    stride: Optional[tuple[int]] = eqx.field(static=True)
    padding: Optional[tuple[int]] = eqx.field(static=True)
    lhs_dilation: Optional[tuple[int]] = eqx.field(static=True)
    rhs_dilation: Optional[tuple[int]] = eqx.field(static=True)
    D: int = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: tuple[tuple[ml.LayerKey, int]],
        target_keys: tuple[tuple[ml.LayerKey, int]],
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
        for (in_k, in_p), in_c in self.input_keys:
            self.weights[(in_k, in_p)] = {}
            for (out_k, out_p), out_c in self.target_keys:
                key, subkey1, subkey2 = random.split(key, num=3)

                filter_key = (in_k + out_k, (in_p + out_p) % 2)
                if filter_key not in self.invariant_filters:
                    continue  # relevant when there isn't an N=3, (0,1) filter

                # unsure if it is this or spatial?
                bound = 1 / jnp.sqrt(in_c * len(self.invariant_filters[filter_key]))
                self.weights[(in_k, in_p)][(out_k, out_p)] = random.uniform(
                    subkey1,
                    shape=(out_c, in_c, len(self.invariant_filters[filter_key])),
                    minval=-bound,
                    maxval=bound,
                )

                if use_bias:
                    # this may get set multiple times, bound could be different but not a huge issue?
                    self.bias[(out_k, out_p)] = random.uniform(
                        subkey2,
                        shape=(out_c,) + (1,) * (self.D + out_k),
                        minval=-bound,
                        maxval=bound,
                    )

    def __call__(self: Self, input_layer: geom.Layer):
        layer = input_layer.empty()
        for (in_k, in_p), images_block in input_layer.items():
            for (out_k, out_p), weight_block in self.weights[(in_k, in_p)].items():
                filter_key = (in_k + out_k, (in_p + out_p) % 2)

                # (out_c,in_c,num_inv_filters) (num, spatial, tensor) -> (out_c,in_c,spatial,tensor)
                filter_block = jnp.einsum(
                    "ijk,k...->ij...", weight_block, self.invariant_filters[filter_key]
                )

                convolve_contracted_imgs = geom.convolve_contract(
                    input_layer.D,
                    images_block[None],  # add batch dim
                    filter_block,
                    input_layer.is_torus,
                    self.stride,
                    self.padding,
                    self.lhs_dilation,
                    self.rhs_dilation,
                )[
                    0
                ]  # remove batch dim

                if (out_k, out_p) in layer:  # it already has that key
                    layer[(out_k, out_p)] = convolve_contracted_imgs + layer[(out_k, out_p)]
                else:
                    layer.append(out_k, out_p, convolve_contracted_imgs)

        if self.use_bias:
            biased_layer = layer.empty()
            for (k, p), image in layer.items():
                if (k, p) == (0, 0) and (self.use_bias == "scalar" or self.use_bias == "auto"):
                    biased_layer.append(k, p, image + self.bias[(k, p)])
                elif ((k, p) != (0, 0) and self.use_bias == "auto") or self.use_bias == "mean":
                    mean_image = jnp.mean(
                        image, axis=tuple(range(1, 1 + self.invariant_filters.D)), keepdims=True
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
    scale: dict[ml.LayerKey, jax.Array]
    bias: dict[ml.LayerKey, jax.Array]
    vanilla_norm: dict[ml.LayerKey, eqx.nn.GroupNorm]

    D: int = eqx.field(static=False)
    groups: int = eqx.field(static=False)
    eps: float = eqx.field(static=False)

    def __init__(
        self: Self,
        input_keys: tuple[tuple[ml.LayerKey, int]],
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
                whitened_data = self.vanilla_norm[(k, p)](image_block)  # do normal group norm
            elif k == 1:
                # save mean vec, allows for un-mean centering (?)
                mean_vec = jnp.mean(image_block, axis=tuple(range(1, 1 + self.D)), keepdims=True)
                assert mean_vec.shape == (len(image_block),) + (1,) * self.D + (self.D,) * k
                whitened_data = ml._group_norm_K1(
                    self.D, image_block[None], self.groups, eps=self.eps
                )
                whitened_data = whitened_data * self.scale[(k, p)] + self.bias * mean_vec
            elif k > 1:
                raise NotImplementedError(
                    f"ml::group_norm: Equivariant group_norm not implemented for k>1, but k={k}",
                )

            out_x.append(k, p, whitened_data)

        return out_x


class LayerNorm(GroupNorm):

    def __init__(
        self: Self, input_keys: tuple[tuple[ml.LayerKey, int]], D: int, eps: float = 1e-5
    ) -> Self:
        super(LayerNorm, self).__init__(input_keys, D, 1, eps)


class VectorNeuronNonlinear(eqx.Module):
    weights: dict[ml.LayerKey, jax.Array]

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

    def __call__(self: Self, x: geom.Layer):
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
    patch_len: int = eqx.field(static=True)
    use_norm: bool = eqx.field(static=True)

    def __init__(self: Self, patch_len: int, use_norm: bool = True):
        self.patch_len = patch_len
        self.use_norm = use_norm

    def __call__(self: Self, x: geom.Layer):
        vmap_max_pool = jax.vmap(geom.max_pool, in_axes=(None, 0, None, None))

        out_x = x.empty()
        for (k, p), image_block in x.items():
            out_x.append(k, p, vmap_max_pool(x.D, image_block, self.patch_len, self.use_norm))

        return out_x


class LayerWrapper(eqx.Module):
    modules: dict[ml.LayerKey, Union[eqx.Module, Callable]]

    def __init__(
        self: Self, module: Union[eqx.Module, Callable], input_keys: tuple[tuple[ml.LayerKey, int]]
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
            out_layer.append(k, p, self.modules[(k, p)](image))

        return out_layer


def save(filename, model):
    # TODO: save batch stats
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


# ~~~~~~~~~~~~~~~~~~~~~~ Training Functions ~~~~~~~~~~~~~~~~~~~~~~
def autoregressive_map(
    batch_model: eqx.Module,
    x: geom.BatchLayer,
    aux_data: Any = None,
    past_steps: int = 1,
    future_steps: int = 1,
    has_aux: bool = False,
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
        if has_aux:
            learned_x, aux_data = batch_model(x, aux_data)
        else:
            learned_x = batch_model(x)

        x, out_x = ml.autoregressive_step(x, learned_x, out_x, past_steps)

    return (out_x, aux_data) if has_aux else out_x


def get_batch_layer(
    layers: Union[Sequence[geom.BatchLayer], geom.BatchLayer],
    batch_size: int,
    rand_key: Optional[ArrayLike],
    sharding: jax.sharding.PositionalSharding,
) -> Union[list[list[geom.BatchLayer]], list[geom.BatchLayer]]:
    """
    Given a set of layers, construct random batches of those layers. The most common use case is for
    layers to be a tuple (X,Y) so that the batches have the inputs and outputs. In this case, it will return
    a list of length 2 where the first element is a list of the batches of the input data and the second
    element is the same batches of the output data.
    args:
        layers (BatchLayer or iterable of BatchLayer): batch layers which all get simultaneously batched
        batch_size (int): length of the batch
        rand_key (jnp random key): key for the randomness. If None, the order won't be random
    returns: list of lists of batches (which are BatchLayers)
    """
    if isinstance(layers, geom.BatchLayer):
        layers = (layers,)

    L = layers[0].get_L()
    batch_indices = jnp.arange(L) if rand_key is None else random.permutation(rand_key, L)

    batches = [[] for _ in range(len(layers))]
    # if L is not divisible by batch, the remainder will be ignored
    for i in range(int(math.floor(L / batch_size))):  # iterate through the batches of an epoch
        idxs = batch_indices[i * batch_size : (i + 1) * batch_size]
        for j, layer in enumerate(layers):
            batches[j].append(layer.get_subset(idxs).device_put(sharding))

    return batches if (len(batches) > 1) else batches[0]


@eqx.filter_jit
def evaluate(
    model: eqx.Module,
    map_and_loss: Union[
        Callable[[eqx.Module, geom.BatchLayer, geom.BatchLayer], jax.Array],
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, Any],
            tuple[jax.Array, Any],
        ],
    ],
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    sharding: jax.sharding.PositionalSharding,
    map_kwargs: dict[str, Any] = {},
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
    replicated = sharding.replicate()
    inference_model, map_kwargs = eqx.filter_shard((inference_model, map_kwargs), replicated)
    x, y = x.device_put(sharding), y.device_put(sharding)
    print(map_kwargs)
    return map_and_loss(inference_model, x, y, **map_kwargs)


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
    map_and_loss: Union[
        Callable[[eqx.Module, geom.BatchLayer, geom.BatchLayer], jax.Array],
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, Any],
            tuple[jax.Array, Any],
        ],
    ],
    model: eqx.Module,
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    batch_size: int,
    rand_key: ArrayLike,
    reducers: Optional[tuple] = None,
    sharding: Optional[jax.sharding.PositionalSharding] = None,
    **map_kwargs: Optional[dict[str, Any]],
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
        batch_size (int): effective batch_size, must be divisible by number of gpus
        rand_key (jax.random.PRNGKey): rand key
        sharding: sharding over multiple GPUs, if None (default), will use available devices
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
    returns: average loss over the entire layer
    """
    inference_model = eqx.nn.inference_mode(model)

    if reducers is None:
        # use the default reducer for loss
        reducers = [loss_reducer]
        if "has_aux" in map_kwargs and map_kwargs["has_aux"]:
            reducers.append(aux_data_reducer)

        if "return_map" in map_kwargs and map_kwargs["return_map"]:
            reducers.append(layer_reducer)

    if sharding is None:
        devices = jax.devices()
        num_devices = len(devices)
        devices = mesh_utils.create_device_mesh((num_devices, 1))
        sharding = jax.sharding.PositionalSharding(devices)

    replicated = sharding.replicate()
    inference_model, map_kwargs = eqx.filter_shard((inference_model, map_kwargs), replicated)

    X_batches, Y_batches = get_batch_layer((x, y), batch_size, rand_key, sharding)
    results = [[] for _ in range(len(reducers))]
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        one_result = evaluate(inference_model, map_and_loss, X_batch, Y_batch, sharding, map_kwargs)

        if len(reducers) == 1:
            results[0].append(one_result)
        else:
            for val, result_ls in zip(one_result, results):
                result_ls.append(val)

    if len(reducers) == 1:
        return reducers[0](results[0])
    else:
        return tuple(reducer(result_ls) for reducer, result_ls in zip(reducers, results))


def map_in_batches(
    map_f: Union[
        Callable[[eqx.Module, geom.BatchLayer, Any], tuple[Any, Any]],
        Callable[[eqx.Module, geom.BatchLayer], Any],
    ],
    batch_model: eqx.Module,
    x: geom.BatchLayer,
    batch_size: int,
    sharding: Optional[jax.sharding.PositionalSharding] = None,
    merge_layer: bool = False,
    has_aux: bool = False,
    aux_data: Optional[Any] = None,
) -> geom.BatchLayer:
    """
    Runs model for the entire x, splitting into batches if it is larger than the batch_size. This
    is helpful to run a whole validation/test set through model when you need to split those over
    batches for memory reasons. Automatically shards over multiple gpus, so the number of gpus
    must evenly divide batch_size as well as as any remainder of the layer.
    args:
        map_f: function with input model and batch layer, and output is some collection of batch layers.
        batch_model (eqx.Module): model that operates on a batch layer
        x (BatchLayer): input data
        batch_size (int): effective batch_size, must be divisible by number of gpus
        has_aux (bool): has auxilliary data, such as batch_stats, defaults to False
        aux_data (any): auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        merge_layer (bool): if the result is a list of layers, automatically merge those layers
    returns: average loss over the entire layer
    """
    inference_model = eqx.nn.inference_mode(batch_model)

    if sharding is None:
        devices = mesh_utils.create_device_mesh((len(jax.devices()), 1))
        sharding = jax.sharding.PositionalSharding(devices)

    replicated = sharding.replicate()
    inference_model, aux_data = eqx.filter_shard((inference_model, aux_data), replicated)

    results = []
    for X_batch in get_batch_layer(x, batch_size, None, sharding):
        print(X_batch)  # TODO: HERE!
        if has_aux:
            batch_out, aux_data = eqx.filter_jit(map_f)(inference_model, X_batch, aux_data)
        else:
            batch_out = eqx.filter_jit(map_f)(inference_model, X_batch)

        results.append(batch_out.device_put(sharding))

    if merge_layer:  # if the results is a list of layers, automatically merge those layers
        return functools.reduce(lambda carry, val: carry.concat(val), results, results[0].empty())
    else:
        return results


@eqx.filter_jit(donate="all")
def train_step(
    map_and_loss: Union[
        Callable[[eqx.Module, geom.BatchLayer, geom.BatchLayer], jax.Array],
        Callable[
            [eqx.Module, geom.BatchLayer, geom.BatchLayer, Any],
            tuple[jax.Array, Any],
        ],
    ],
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state,
    x: geom.BatchLayer,
    y: geom.BatchLayer,
    sharding: jax.sharding.PositionalSharding,
    has_aux: bool = False,
    aux_data: Optional[Any] = None,
) -> tuple[eqx.Module, Any, Union[float, tuple[float, Any]]]:
    """
    Perform one step and gradient update of the model. If has_aux=True, then the return loss_value
    is a tuple of (loss, aux_data).
    args:
        map_and_loss (func): map and loss function where the input is a model pytree, x BatchLayer,
            y BatchLayer, and aux_data, and returns a float loss (and aux_data if its used)
        model (equinox model pytree): the model
        optim (optax optimizer):
        opt_state:
        x (BatchLayer): input data
        y (BatchLayer): target data
        sharding (PositionalSharding): jax GPU sharding
        has_aux (bool): whether the data has stateful layers like BatchNorm
        aux_data (Any): auxilliary data for stateful layers
    returns: model, opt_state, loss_value
    """
    loss_grad = eqx.filter_value_and_grad(map_and_loss, has_aux=has_aux)

    # do sharding for multiple GPUs
    replicated = sharding.replicate()
    model, opt_state, aux_data = eqx.filter_shard((model, opt_state, aux_data), replicated)
    x, y = x.device_put(sharding), y.device_put(sharding)

    args = (model, x, y, aux_data) if has_aux else (model, x, y)
    loss_value, grads = loss_grad(*args)
    if has_aux:
        loss_value, aux_data = loss_value

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model, opt_state, aux_data = eqx.filter_shard((model, opt_state, aux_data), replicated)
    return model, opt_state, loss_value, aux_data


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
    stop_condition: ml.StopCondition,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    validation_X: Optional[geom.BatchLayer] = None,
    validation_Y: Optional[geom.BatchLayer] = None,
    save_model: Optional[str] = None,
    has_aux: bool = False,
    aux_data: Optional[Any] = None,
    devices: Optional[list[jax.Device]] = None,
) -> Union[tuple[eqx.Module, Any, jax.Array, jax.Array], tuple[eqx.Module, jax.Array, jax.Array]]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    shards over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    args:
        X (BatchLayer): The X input data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        Y (BatchLayer): The Y target data as a layer by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss (function): function that takes in params, X_batch, Y_batch, rand_key, and train and
            returns the loss. If has_aux is True, then it also takes in aux_data and returns aux_data.
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
        has_aux (bool): Passed to value_and_grad, specifies whether there is auxilliary data returned from
            map_and_loss. If true, this auxilliary data will be passed back in to map_and_loss with the
            name "aux_data". The last aux_data will also be returned from this function.
        aux_data (any): initial aux data passed in to map_and_loss when has_aux is true.
        devices (list): gpu/cpu devices to use, if None (default) then it will use jax.devices()
    returns: A tuple of best model in inference mode, epoch loss, and val loss
    """
    if isinstance(stop_condition, ml.ValLoss) and not (validation_X and validation_Y):
        raise ValueError("Stop condition is ValLoss, but no validation data provided.")

    devices = devices if devices else jax.devices()
    num_devices = len(devices)
    devices = mesh_utils.create_device_mesh((num_devices, 1))
    sharding = jax.sharding.PositionalSharding(devices)
    replicated = sharding.replicate()
    model, aux_data = eqx.filter_shard((model, aux_data), replicated)

    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    val_loss = 0
    epoch_time = 0
    while not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, epoch_time):
        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batch_layer((X, Y), batch_size, subkey, sharding)
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
                sharding,
                has_aux,
                aux_data,
            )
            epoch_loss += jnp.mean(loss_value)

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
                sharding=sharding,
                has_aux=has_aux,
                aux_data=aux_data,
            )
            val_loss = epoch_val_loss

        if save_model and ((epoch % 10) == 0):
            save(save_model, model)

        epoch_time = time.time() - start_time

    if has_aux:
        return (
            stop_condition.best_params,
            aux_data,
            epoch_loss,
            val_loss,
        )
    else:
        return eqx.nn.inference_mode(stop_condition.best_params), epoch_loss, val_loss
