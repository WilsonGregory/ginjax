import time
import math
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
    weights: dict[ml.LayerKey, dict[ml.LayerKey, float]]
    bias: dict[ml.LayerKey, float]

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


# ~~~~~~~~~~~~~~~~~~~~~~ Training Functions ~~~~~~~~~~~~~~~~~~~~~~
def get_batch_layer(
    layers: Union[Sequence[geom.BatchLayer], geom.BatchLayer],
    batch_size: int,
    rand_key: ArrayLike,
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
    sharding: Optional[jax.sharding.PositionalSharding] = None,
    has_aux: bool = False,
    aux_data: Optional[Any] = None,
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

    if sharding is None:
        devices = jax.devices()
        num_devices = len(devices)
        devices = mesh_utils.create_device_mesh((num_devices, 1))
        sharding = jax.sharding.PositionalSharding(devices)

    replicated = sharding.replicate()
    inference_model = eqx.filter_shard(inference_model, replicated)

    rand_key, subkey = random.split(rand_key)
    X_batches, Y_batches = get_batch_layer((x, y), batch_size, subkey, sharding)
    total_loss = None
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        if has_aux:
            one_loss, aux_data = map_and_loss(inference_model, X_batch, Y_batch, aux_data)
        else:
            one_loss = map_and_loss(inference_model, X_batch, Y_batch)

        total_loss = (0 if total_loss is None else total_loss) + one_loss

    return total_loss / len(X_batches)


@eqx.filter_jit(donate="all")
def make_step(
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
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
    x, y = x.device_put(sharding), y.device_put(sharding)

    args = (model, x, y, aux_data) if has_aux else (model, x, y)
    loss_value, grads = loss_grad(*args)
    if has_aux:
        loss_value, aux_data = loss_value

    updates, opt_state = optim.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    model, opt_state = eqx.filter_shard((model, opt_state), replicated)
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
    model = eqx.filter_shard(model, replicated)

    opt_state = optimizer.init(model)
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
            model, opt_state, loss_value, aux_data = make_step(
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
                sharding,
                has_aux,
                aux_data,
            )
            val_loss = epoch_val_loss

        if save_model and ((epoch % 10) == 0):
            raise NotImplementedError("train::Saving a model is not implemented yet")

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
