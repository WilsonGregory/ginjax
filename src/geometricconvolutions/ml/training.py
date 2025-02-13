import time
import math
import functools
from typing import Any, Callable, Optional, Sequence, Union
from typing_extensions import Self
import numpy as np

import jax
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike
import equinox as eqx
import optax

import geometricconvolutions.geometric as geom
from geometricconvolutions.ml.stopping_conditions import StopCondition, ValLoss


def save(filename, model):
    # TODO: save batch stats
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def load(filename, model):
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


## Data and Batching operations


def get_batches(
    multi_images: Union[Sequence[geom.BatchMultiImage], geom.BatchMultiImage],
    batch_size: int,
    rand_key: ArrayLike,
    devices: Optional[list[jax.Device]] = None,
) -> list[list[geom.BatchMultiImage]]:
    """
    Given a set of MultiImages, construct random batches of those MultiImages. The most common use case
    is for MultiImagess to be a tuple (X,Y) so that the batches have the inputs and outputs. In this case, it will return
    a list of length 2 where the first element is a list of the batches of the input data and the second
    element is the same batches of the output data. Automatically reshapes the batches to use with
    pmap based on the number of gpus found.
    args:
        multi_images: BatchMultiImages which all get simultaneously batched
        batch_size: length of the batch
        rand_key: key for the randomness. If None, the order won't be random
        devices: gpu/cpu devices to use, if None (default) then sets this to jax.devices()
    returns: list of lists of batches (which are BatchMultiImages)
    """
    if isinstance(multi_images, geom.BatchMultiImage):
        multi_images = (multi_images,)

    L = multi_images[0].get_L()
    batch_indices = jnp.arange(L) if rand_key is None else random.permutation(rand_key, L)

    if devices is None:
        devices = jax.devices()

    batches = [[] for _ in range(len(multi_images))]
    # if L is not divisible by batch, the remainder will be ignored
    for i in range(int(math.floor(L / batch_size))):  # iterate through the batches of an epoch
        idxs = batch_indices[i * batch_size : (i + 1) * batch_size]
        for j, multi_image in enumerate(multi_images):
            batches[j].append(multi_image.get_subset(idxs).reshape_pmap(devices))

    return batches


# ~~~~~~~~~~~~~~~~~~~~~~ Training Functions ~~~~~~~~~~~~~~~~~~~~~~


def autoregressive_step(
    input: geom.BatchMultiImage,
    one_step: geom.BatchMultiImage,
    output: geom.BatchMultiImage,
    past_steps: int,
    constant_fields: dict[tuple[int, int], int] = {},
    future_steps: int = 1,
) -> tuple[geom.BatchMultiImage, geom.BatchMultiImage]:
    """
    Given the input MultiImage, the next step of the model, and the output, update the input
    and output to be fed into the model next. BatchMultiImages should have shape (batch,channels,spatial,tensor).
    Channels are c*past_steps + constant_steps where c is some positive integer.
    args:
        input: the input to the model
        one_step: the model output at this step, assumed to be a single time step
        output: the full output that we are building up
        past_steps: the number of past time steps that are fed into the model
        constant_fields: a map {key:n_constant_fields} for fields that don't depend on timestep
        future_steps: number of future steps that the model outputs, currently must be 1
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
    x: geom.BatchMultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    past_steps: int = 1,
    future_steps: int = 1,
) -> tuple[geom.BatchMultiImage, Optional[eqx.nn.State]]:
    """
    Given a model, perform an autoregressive step (future_steps) times, and return the output
    steps in a single BatchMultiImage.
    args:
        batch_model: model that operates of batches, probably a vmapped version of model.
        x: the input MultiImage to map
        past_steps: the number of past steps input to the autoregressive map, default 1
        future_steps: how many times to loop through the autoregression, default 1
        aux_data: auxilliary data to pass to the network
        has_aux: whether net returns an aux_data, defaults to False
    """
    assert callable(batch_model)
    out_x = x.empty()  # assume out matches D and is_torus
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
            [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
            tuple[jax.Array, Optional[eqx.nn.State]],
        ],
        Callable[
            [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
            tuple[jax.Array, Optional[eqx.nn.State], geom.BatchMultiImage],
        ],
    ],
    x: geom.BatchMultiImage,
    y: geom.BatchMultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> Union[jax.Array, tuple[jax.Array, geom.BatchMultiImage]]:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the BatchMultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the BatchMultiImage.

    args:
        model: the model to run through map_and_loss
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        x: input data
        y: target output data
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        return_map: whether to also return the map of x

    Returns:
        Average loss over the entire BatchMultiImage
    """
    inference_model = eqx.nn.inference_mode(model)
    if return_map:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None, 0),
        )
        loss, _, out = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0), out.merge_pmap()
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


def multi_image_reducer(ls):
    """
    If map data returns the mapped MultiImages, merge them togther
    """
    return functools.reduce(lambda carry, val: carry.concat(val), ls, ls[0].empty())


def map_loss_in_batches(
    map_and_loss: Callable[
        [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: eqx.Module,
    x: geom.BatchMultiImage,
    y: geom.BatchMultiImage,
    batch_size: int,
    rand_key: ArrayLike,
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> jax.Array:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the BatchMultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the BatchMultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        x: input data
        y: target output data
        batch_size: effective batch_size, must be divisible by number of gpus
        rand_key: rand key
        devices: the gpus that the code will run on
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.

    Returns:
        Average loss over the entire BatchMultiImage
    """
    X_batches, Y_batches = get_batches((x, y), batch_size, rand_key, devices)
    losses = [
        evaluate(model, map_and_loss, X_batch, Y_batch, aux_data, False)
        for X_batch, Y_batch in zip(X_batches, Y_batches)
    ]
    return loss_reducer(losses)


def map_plus_loss_in_batches(
    map_and_loss: Callable[
        [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State], geom.BatchMultiImage],
    ],
    model: eqx.Module,
    x: geom.BatchMultiImage,
    y: geom.BatchMultiImage,
    batch_size: int,
    rand_key: ArrayLike,
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[jax.Array, geom.BatchMultiImage]:
    """
    This is like `map_loss_in_batches`, but it returns the mapped images in additon to just the loss.
    Runs map_and_loss for the entire x, y, splitting into batches if the BatchMultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the BatchMultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        x: input data
        y: target output data
        batch_size: effective batch_size, must be divisible by number of gpus
        rand_key: rand key
        devices: the gpus that the code will run on
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.

    Returns:
        Average loss over the entire BatchMultiImage, and the mapped entire BatchMultiImage
    """
    X_batches, Y_batches = get_batches((x, y), batch_size, rand_key, devices)
    losses = []
    out_maps = []
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        one_loss, one_map = evaluate(model, map_and_loss, X_batch, Y_batch, aux_data, True)

        losses.append(one_loss)
        out_maps.append(one_map)

    return loss_reducer(losses), multi_image_reducer(out_maps)


def train_step(
    map_and_loss: Callable[
        [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: eqx.Module,
    optim: optax.GradientTransformation,
    opt_state,
    x: geom.BatchMultiImage,
    y: geom.BatchMultiImage,
    aux_data: Optional[eqx.nn.State] = None,
):
    """
    Perform one step and gradient update of the model. Uses filter_pmap to use multiple gpus.
    args:
        map_and_loss: map and loss function where the input is a model pytree, x, y, and
            aux_data, and returns a float loss and aux_data
        model (equinox model pytree): the model
        optim:
        opt_state:
        x: input data
        y: target data
        aux_data: auxilliary data for stateful layers
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
    X: geom.BatchMultiImage,
    Y: geom.BatchMultiImage,
    map_and_loss: Callable[
        [eqx.Module, geom.BatchMultiImage, geom.BatchMultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: eqx.Module,
    rand_key: ArrayLike,
    stop_condition: StopCondition,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    validation_X: Optional[geom.BatchMultiImage] = None,
    validation_Y: Optional[geom.BatchMultiImage] = None,
    save_model: Optional[str] = None,
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[eqx.Module, Optional[eqx.nn.State], Optional[ArrayLike], Optional[ArrayLike]]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    shards over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    args:
        X: The X input data as a BatchMultiImage by k of (images, channels, (N,)*D, (D,)*k)
        Y: The Y target data as a BatchMultiImage by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss: function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        rand_key: key for randomness
        stop_condition: when to stop the training process, currently only 1 condition
            at a time
        batch_size: the size of each mini-batch in SGD
        optimizer: optimizer
        validation_X: input data for a validation data set as a BatchMultiImage by k
            of (images, channels, (N,)*D, (D,)*k)
        validation_Y: target data for a validation data set as a BatchMultiImage by k
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
    val_loss = None
    epoch_time = 0
    stop_condition.best_model = model
    while not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, epoch_time):
        rand_key, subkey = random.split(rand_key)
        X_batches, Y_batches = get_batches((X, Y), batch_size, subkey, devices)
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
    get_data: Callable[[Any], tuple[geom.BatchMultiImage, ...]],
    models: list[tuple[str, Callable]],
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
