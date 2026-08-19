import time
import enum
import json
import math
import functools
import pathlib
from typing import Any, Callable, Optional, Sequence, Union
from typing_extensions import Self
import numpy as np
import wandb

import jax
import jax.numpy as jnp
import jax.random as random
from jax.typing import ArrayLike
import equinox as eqx
import optax
from torch.utils.data import BatchSampler, DataLoader, Dataset, RandomSampler, SequentialSampler

import ginjax.geometric as geom
from ginjax.ml.stopping_conditions import StopCondition, ValLoss
from ginjax.ml.losses import (
    smse_loss,
    nrmse_loss,
    nrmse_per_pixel_loss,
    l2_rel_error,
    l2_per_pixel_rel_error,
)
import ginjax.models as models


def save(filename: str | pathlib.Path, model: models.MultiImageModule) -> None:
    """
    Save an equinox model.

    args:
        filename: the file to save the model to
        model: the model to save
    """
    # TODO: save batch stats
    with open(filename, "wb") as f:
        eqx.tree_serialise_leaves(f, model)


def save_plus(
    filename: str | pathlib.Path, model: models.MultiImageModule, further_args: dict = {}
) -> None:
    """
    New version of save, allows you to save any serializable args.

    args:
        filename: the file to save the model to
        model: the model to save
        further_args: more values to save, as a dictionary
    """
    # TODO: save batch stats
    with open(filename, "wb") as f:
        further_args_str = json.dumps(further_args)
        f.write((further_args_str + "\n").encode())
        eqx.tree_serialise_leaves(f, model)


def load(filename: str | pathlib.Path, model: models.MultiImageModule) -> models.MultiImageModule:
    """
    Load an equinox model.

    args:
        filename: the file to load the model from
        model: the type of model we are loading, the parameter values will be set to the loaded ones

    returns:
        the loaded model
    """
    with open(filename, "rb") as f:
        return eqx.tree_deserialise_leaves(f, model)


def load_plus(
    filename: str | pathlib.Path, model: models.MultiImageModule
) -> tuple[models.MultiImageModule, dict]:
    """
    Load an equinox model.

    args:
        filename: the file to load the model from
        model: the type of model we are loading, the parameter values will be set to the loaded ones

    returns:
        the loaded model
    """
    with open(filename, "rb") as f:
        further_args = json.loads(f.readline().decode())
        return eqx.tree_deserialise_leaves(f, model), further_args


## Data and Batching operations


class MultiImageDataset(Dataset):
    """
    A basic dataset for multi images which assumes that we already have the X and Y in memory.
    The __getitem__ for this class expects a list of integer indices for the entire batch at
    once, which means the sampler of the data loader should be a batch sampler.
    """

    D: int
    X: geom.MultiImage
    Y: geom.MultiImage
    devices: list[jax.Device]
    use_devices: bool

    def __init__(
        self: Self,
        X: geom.MultiImage,
        Y: geom.MultiImage,
        devices: list[jax.Device] | None = None,
        use_devices: bool = True,
    ) -> None:
        self.D = X.D
        self.X = X
        self.Y = Y
        self.devices = devices if devices else jax.devices()
        self.use_devices = use_devices

    def __len__(self: Self) -> int:
        return self.X.get_L()

    def __getitem__(self: Self, idx: list[int]) -> tuple[geom.MultiImage, geom.MultiImage]:
        idxs = jnp.array(idx)
        X_batch, Y_batch = self.X.get_subset(idxs), self.Y.get_subset(idxs)

        if self.use_devices:
            X_batch = X_batch.reshape_pmap(self.devices)
            Y_batch = Y_batch.reshape_pmap(self.devices)

        return X_batch, Y_batch

    def get_N(self: Self) -> int:
        return self.X.get_spatial_dims()[0]


def get_batches(
    multi_images: Union[Sequence[geom.MultiImage], geom.MultiImage],
    batch_size: int,
    rand_key: Optional[ArrayLike],
    devices: Optional[list[jax.Device]] = None,
) -> list[list[geom.MultiImage]]:
    """
    Given a set of MultiImages, construct random batches of those MultiImages. The most common use case
    is for MultiImagess to be a tuple (X,Y) so that the batches have the inputs and outputs. In this case, it will return
    a list of length 2 where the first element is a list of the batches of the input data and the second
    element is the same batches of the output data. Automatically reshapes the batches to use with
    pmap based on the number of gpus found.

    args:
        multi_images: MultiImages which all get simultaneously batched
        batch_size: length of the batch
        rand_key: key for the randomness. If None, the order won't be random
        devices: gpu/cpu devices to use, if None (default) then sets this to jax.devices()

    returns:
        list of lists of batches (which are MultiImages)
    """
    if isinstance(multi_images, geom.MultiImage):
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
    input: geom.MultiImage,
    output: geom.MultiImage,
    past_steps: int,
    constant_fields_dict: dict[tuple[tuple[bool, ...], int], int] = {},
    future_steps: int = 1,
) -> geom.MultiImage:
    """
    Given the input MultiImage, the next step of the model, update the input to be fed into the
    model next. MultiImages should have shape (channels,spatial,tensor). Channels are
    c*past_steps + constant_fields where c is some positive integer.

    args:
        input: the input to the model
        output: the model output at this step, assumed to be a single time step
        past_steps: the number of past time steps that are fed into the model
        constant_fields_dict: a map {key:n_constant_fields} for fields that don't depend on timestep
        future_steps: number of future steps that the model outputs, currently must be 1

    returns:
        the new input
    """
    assert (
        future_steps == 1
    ), f"ml::autoregressive_step: future_steps must be 1, but found {future_steps}."

    dynamic_input, constant_fields = input.concat_inverse(constant_fields_dict)
    dynamic_input = dynamic_input.expand(0, past_steps)
    output = output.expand(0, future_steps)

    new_input = input.empty()
    for k, parity in input.keys():
        # its important to insert the keys in the same order
        if (k, parity) in dynamic_input:
            assert (k, parity) in output

            # (c,past_steps,spatial,tensor)
            new_input_image = jnp.concatenate(
                [dynamic_input[(k, parity)][:, future_steps:], output[(k, parity)]], axis=1
            )
            # (c*past_steps,spatial,tensor)
            new_input_image = new_input_image.reshape((-1,) + new_input_image.shape[2:])
            new_input.append(k, parity, new_input_image)

        if (k, parity) in constant_fields:
            new_input.append(k, parity, constant_fields[(k, parity)])

    return new_input


def autoregressive_map(
    model: models.MultiImageModule,
    x: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    past_steps: int = 1,
    autoregressive_steps: int = 1,
    constant_fields: dict[tuple[tuple[bool, ...], int], int] = {},
) -> tuple[geom.MultiImage, Optional[eqx.nn.State]]:
    """
    Given a model, perform an autoregressive step n times, and return the output
    steps in a single MultiImage. Currently the model must output a single time step.

    args:
        model: model that operates on MultiImages
        x: the input MultiImage to map
        aux_data: auxilliary data to pass to the network
        past_steps: the number of past steps input to the autoregressive map
        autoregressive_steps: how many times to loop through the autoregression
        constant_fields: data structure which explains which fields are constant fields

    returns:
        the output map with number of steps equal to future steps, and the aux_data
    """
    future_steps = 1
    out_x = x.empty()  # assume out matches D and is_torus
    for _ in range(autoregressive_steps):
        pred_x, aux_data = model(x, aux_data)
        x = autoregressive_step(x, pred_x, past_steps, constant_fields)

        out_x = out_x.concat(pred_x.expand(axis=0, size=future_steps), axis=1)

    return out_x.combine_axes((0, 1)), aux_data


def evaluate(
    model: models.MultiImageModule,
    map_and_loss: Union[
        Callable[
            [models.MultiImageModule, geom.MultiImage, geom.MultiImage, Optional[eqx.nn.State]],
            tuple[jax.Array, Optional[eqx.nn.State]],
        ],
        Callable[
            [models.MultiImageModule, geom.MultiImage, geom.MultiImage, Optional[eqx.nn.State]],
            tuple[jax.Array, Optional[eqx.nn.State], geom.MultiImage],
        ],
    ],
    x: geom.MultiImage,
    y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
    return_map: bool = False,
) -> Union[jax.Array, tuple[jax.Array, geom.MultiImage]]:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the MultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the MultiImage.

    args:
        model: the model to run through map_and_loss
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data if has_aux is true, and returns the loss, and aux_data if has_aux is true.
        x: input data
        y: target output data
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        return_map: whether to also return the map of x

    Returns:
        Average loss over the entire MultiImage
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
        return jnp.mean(loss, axis=0), out.merge_axes([0, 1])
    else:
        compute_loss_pmap = eqx.filter_pmap(
            map_and_loss,
            axis_name="pmap_batch",
            in_axes=(None, 0, 0, None),
            out_axes=(0, None),
        )
        loss, _ = compute_loss_pmap(inference_model, x, y, aux_data)
        return jnp.mean(loss, axis=0)


def loss_reducer(ls: list[jax.Array]) -> jax.Array:
    """
    A reducer for map_loss_in_batches that takes the batch mean of the loss

    args:
        ls: list of losses

    returns:
        the mean of the losses
    """
    return jnp.mean(jnp.stack(ls), axis=0)


def multi_image_reducer(ls: list[geom.MultiImage]) -> geom.MultiImage:
    """
    If map data returns the mapped MultiImages, merge them togther

    args:
        ls: list of MultiImages

    returns:
        a single concatenated MultiImage
    """
    return functools.reduce(lambda carry, val: carry.concat(val), ls, ls[0].empty())


def map_loss_in_batches_dl(
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
        tuple[jax.Array, eqx.nn.State | None],
    ],
    model: models.MultiImageModule,
    dataloader: DataLoader,
    aux_data: eqx.nn.State | None = None,
    reduce: str | None = "mean",
) -> jax.Array:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the MultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the MultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        dataloader: the dataloader for input and output multi image data
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        reduce: how to reduce between batches, defaults to mean

    Returns:
        Average loss over the entire BatchMultiImage
    """
    losses = []
    for X_batch, Y_batch in dataloader:
        losses.append(evaluate(model, map_and_loss, X_batch, Y_batch, aux_data, False))

    return loss_reducer(losses) if reduce == "mean" else jnp.concat(losses, axis=0)


def map_loss_in_batches(
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: models.MultiImageModule,
    x: geom.MultiImage,
    y: geom.MultiImage,
    batch_size: int,
    rand_key: Optional[ArrayLike],
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
    reduce: str | None = "mean",
) -> jax.Array:
    """
    Runs map_and_loss for the entire x, y, splitting into batches if the MultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the MultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        x: input data
        y: target output data
        batch_size: effective batch_size, must be divisible by number of gpus
        rand_key: rand key passed to get_batches, on None order won't be randomized
        devices: the gpus that the code will run on
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.
        reduce: how to reduce between batches, defaults to mean

    Returns:
        Average loss over the entire BatchMultiImage
    """
    dataset = MultiImageDataset(x, y, devices)
    dataloader = DataLoader(
        dataset,
        sampler=BatchSampler(SequentialSampler(dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )
    return map_loss_in_batches_dl(map_and_loss, model, dataloader, aux_data, reduce)


def map_plus_loss_in_batches(
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State], geom.MultiImage],
    ],
    model: models.MultiImageModule,
    x: geom.MultiImage,
    y: geom.MultiImage,
    batch_size: int,
    rand_key: Optional[ArrayLike],
    devices: Optional[list[jax.Device]] = None,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[jax.Array, geom.MultiImage]:
    """
    This is like `map_loss_in_batches`, but it returns the mapped images in additon to just the loss.
    Runs map_and_loss for the entire x, y, splitting into batches if the MultiImage is larger than
    the batch_size. This is helpful to run a whole validation/test set through map and loss when you need
    to split those over batches for memory reasons. Automatically pmaps over multiple gpus, so the number
    of gpus must evenly divide batch_size as well as as any remainder of the MultiImage.

    args:
        map_and_loss: function that takes in model, X_batch, Y_batch, and
            aux_data and returns the loss and aux_data
        model: the model to run through map_and_loss
        x: input data
        y: target output data
        batch_size: effective batch_size, must be divisible by number of gpus
        rand_key: rand key passed to get_batches, on none the order will not be randomized
        devices: the gpus that the code will run on
        aux_data: auxilliary data, such as batch stats. Passed to the function is has_aux is True.

    Returns:
        Average loss over the entire MultiImage, and the mapped entire MultiImage
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
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, Optional[eqx.nn.State]],
        tuple[jax.Array, Optional[eqx.nn.State]],
    ],
    model: models.MultiImageModule,
    optim: optax.GradientTransformation,
    opt_state: Any,
    x: geom.MultiImage,
    y: geom.MultiImage,
    aux_data: Optional[eqx.nn.State] = None,
) -> tuple[models.MultiImageModule, Any, jax.Array, Optional[eqx.nn.State]]:
    """
    Perform one step and gradient update of the model. Uses filter_pmap to use multiple gpus.

    args:
        map_and_loss: map and loss function where the input is a model pytree, x, y, and
            aux_data, and returns a float loss and aux_data
        model: the model
        optim: the optimizer
        opt_state:
        x: input data
        y: target data
        aux_data: auxilliary data for stateful layers

    returns:
        model, opt_state, loss_value, aux_data
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


def train_dl_wandb(
    train_dataloader: DataLoader,
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
        tuple[jax.Array, eqx.nn.State | None],
    ],
    model: models.MultiImageModule,
    stop_condition: StopCondition,
    optimizer: optax.GradientTransformation,
    val_dataloader: DataLoader | None = None,
    val_map_and_loss: (
        Callable[
            [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
            tuple[jax.Array, eqx.nn.State | None],
        ]
        | None
    ) = None,
    save_model: str | None = None,
    aux_data: eqx.nn.State | None = None,
    is_wandb: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    model_name: str = "",
    args: dict = {},
) -> tuple[models.MultiImageModule, eqx.nn.State | None, ArrayLike | None, ArrayLike | None, float]:
    """
    A wrapper around train_dl that initializes and wandb, runs train_dl, and tears down wandb. If
    is_wandb is false, this function is the same as train_dl.

    args:
        train_dataloader: dataloader for train input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        map_and_loss: function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        stop_condition: when to stop the training process, currently only 1 condition
            at a time
        batch_size: the size of each mini-batch in SGD
        optimizer: optimizer
        val_dataloader: dataloader for val input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        save_model: if string, save model every 10 epochs, defaults to None
        aux_data: initial aux data passed in to map_and_loss when has_aux is true.
        is_wandb: whether wandb experiment tracking has been initiated and should be logged to

    returns:
        A tuple of best model in inference mode, aux_data, epoch loss, val loss, and train_time
    """
    if is_wandb:
        wandb.init(
            project=wandb_project,
            entity=wandb_entity,
            name=model_name,
            settings=wandb.Settings(start_method="fork"),
        )
        type_list = [str, int, float, bool]

        def display_val(val):
            if isinstance(val, enum.Enum):
                return val.name
            elif type(val) in type_list or val is None:
                return val
            else:
                return type(val)

        wandb.config.update(
            {key: display_val(val) for key, val in args.items()},
            allow_val_change=True,
        )

    out = train_dl(
        train_dataloader,
        map_and_loss,
        model,
        stop_condition,
        optimizer,
        val_dataloader,
        val_map_and_loss,
        save_model,
        aux_data,
        is_wandb,
    )

    if is_wandb:
        wandb.finish()

    return out


def train_dl(
    train_dataloader: DataLoader,
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
        tuple[jax.Array, eqx.nn.State | None],
    ],
    model: models.MultiImageModule,
    stop_condition: StopCondition,
    optimizer: optax.GradientTransformation,
    val_dataloader: DataLoader | None = None,
    val_map_and_loss: (
        Callable[
            [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
            tuple[jax.Array, eqx.nn.State | None],
        ]
        | None
    ) = None,
    save_model: str | None = None,
    aux_data: eqx.nn.State | None = None,
    is_wandb: bool = False,
) -> tuple[models.MultiImageModule, eqx.nn.State | None, ArrayLike | None, ArrayLike | None, float]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    pmaps over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    This version uses pytorch datasets and dataloaders.

    args:
        train_dataloader: dataloader for train input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        map_and_loss: function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        stop_condition: when to stop the training process, currently only 1 condition
            at a time
        batch_size: the size of each mini-batch in SGD
        optimizer: optimizer
        val_dataloader: dataloader for val input and target data. Each is a MultiImage by k of
            (images, channels, (N,)*D, (D,)*k)
        save_model: if string, save model every 10 epochs, defaults to None
        aux_data: initial aux data passed in to map_and_loss when has_aux is true.
        is_wandb: whether wandb experiment tracking has been initiated and should be logged to

    returns:
        A tuple of best model in inference mode, aux_data, epoch loss, val loss, and train_time
    """
    if isinstance(stop_condition, ValLoss) and val_dataloader is None:
        raise ValueError("Stop condition is ValLoss, but no validation data provided.")

    if val_map_and_loss is None:
        val_map_and_loss = map_and_loss

    total_train_time = 0
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    epoch = 0
    epoch_val_loss = None
    epoch_loss = None
    val_loss = None
    epoch_time = 0
    stop_condition.best_model = model
    while not stop_condition.stop(model, epoch, epoch_loss, epoch_val_loss, epoch_time):
        start_time = time.time()
        epoch_loss = None
        n_batches = 0
        for X_batch, Y_batch in train_dataloader:
            n_batches += 1
            model, opt_state, loss_value, aux_data = train_step(
                map_and_loss,
                model,
                optimizer,
                opt_state,
                X_batch,
                Y_batch,
                aux_data,
            )
            epoch_loss = loss_value if epoch_loss is None else epoch_loss + loss_value

        total_train_time += time.time() - start_time

        if epoch_loss is not None:
            epoch_loss = epoch_loss / n_batches

        epoch += 1
        log = {"train/loss": epoch_loss}

        # We evaluate the validation loss in batches for memory reasons.
        if val_dataloader is not None:
            epoch_val_loss = map_loss_in_batches_dl(
                val_map_and_loss,
                model,
                val_dataloader,
                aux_data=aux_data,
            )
            val_loss = epoch_val_loss
            log["val/loss"] = val_loss

        if is_wandb:
            wandb.log(log)

        if save_model and ((epoch % 10) == 0):
            save(save_model, model)

        epoch_time = time.time() - start_time

    return stop_condition.best_model, aux_data, epoch_loss, val_loss, total_train_time


def train(
    X: geom.MultiImage,
    Y: geom.MultiImage,
    map_and_loss: Callable[
        [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
        tuple[jax.Array, eqx.nn.State | None],
    ],
    model: models.MultiImageModule,
    rand_key: jax.Array,
    stop_condition: StopCondition,
    batch_size: int,
    optimizer: optax.GradientTransformation,
    validation_X: geom.MultiImage | None = None,
    validation_Y: geom.MultiImage | None = None,
    val_map_and_loss: (
        Callable[
            [models.MultiImageModule, geom.MultiImage, geom.MultiImage, eqx.nn.State | None],
            tuple[jax.Array, eqx.nn.State | None],
        ]
        | None
    ) = None,
    save_model: str | None = None,
    devices: list[jax.Device] | None = None,
    aux_data: eqx.nn.State | None = None,
    is_wandb: bool = False,
) -> tuple[models.MultiImageModule, eqx.nn.State | None, ArrayLike | None, ArrayLike | None, float]:
    """
    Method to train the model. It uses stochastic gradient descent (SGD) with the optimizer to learn the
    parameters the minimize the map_and_loss function. The model is returned. This function automatically
    pmaps over the available gpus, so batch_size should be divisible by the number of gpus. If you only want
    to train on a single GPU, the script should be run with CUDA_VISIBLE_DEVICES=# for whatever gpu number.
    Use train_dl if you would like to pass pytorch dataloaders.

    args:
        X: The X input data as a MultiImage by k of (images, channels, (N,)*D, (D,)*k)
        Y: The Y target data as a MultiImage by k of (images, channels, (N,)*D, (D,)*k)
        map_and_loss: function that takes in model, X_batch, Y_batch, and aux_data and
            returns the loss and aux_data.
        model: Model pytree
        rand_key: key for randomness
        stop_condition: when to stop the training process, currently only 1 condition
            at a time
        batch_size: the size of each mini-batch in SGD
        optimizer: optimizer
        validation_X: input data for a validation data set as a MultiImage by k
            of (images, channels, (N,)*D, (D,)*k)
        validation_Y: target data for a validation data set as a MultiImage by k
            of (images, channels, (N,)*D, (D,)*k)
        save_model: if string, save model every 10 epochs, defaults to None
        aux_data: initial aux data passed in to map_and_loss when has_aux is true.
        devices: gpu/cpu devices to use, if None (default) then it will use jax.devices()
        is_wandb: whether wandb experiment tracking has been initiated and should be logged to

    returns:
        A tuple of best model in inference mode, aux_data, epoch loss, val loss, and train_time
    """
    train_dataset = MultiImageDataset(X, Y, devices)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=BatchSampler(RandomSampler(train_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    if validation_X is not None and validation_Y is not None:
        val_dataset = MultiImageDataset(validation_X, validation_Y, devices)
        val_dataloader = DataLoader(
            val_dataset,
            sampler=BatchSampler(SequentialSampler(val_dataset), batch_size, drop_last=True),
            collate_fn=lambda x: x[0],
        )
    else:
        val_dataloader = None

    return train_dl(
        train_dataloader,
        map_and_loss,
        model,
        stop_condition,
        optimizer,
        val_dataloader,
        val_map_and_loss,
        save_model,
        aux_data,
        is_wandb,
    )


BENCHMARK_DATA = "benchmark_data"
BENCHMARK_MODEL = "benchmark_model"
BENCHMARK_NONE = "benchmark_none"


def benchmark(
    get_data: Callable,
    models: list[tuple[str, Callable, dict]],
    rand_key: ArrayLike,
    benchmark: str,
    benchmark_range: Sequence,
    benchmark_type: str = BENCHMARK_DATA,
    num_trials: int = 1,
    num_results: int = 1,
    is_wandb: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    args: dict = {},
) -> np.ndarray:
    """
    Method to benchmark multiple models as a particular benchmark over the specified range.

    args:
        get_data: function that takes as its first argument the benchmark_value, and a rand_key
            as its second argument. It returns the data which later gets passed to model.
        models: the elements of the tuple are (str) model_name, (func) model, and a dict of keyword
            arguments to pass to model. Model is a function that takes data, a rand_key, the
            model_name, and remaining keyword arguments and returns either a single float score
            or an iterable of length num_results of float scores.
        rand_key: key for randomness
        benchmark: the type of benchmarking to do
        benchmark_range: iterable of the benchmark values to range over
        benchmark_type: one of { BENCHMARK_DATA, BENCHMARK_MODEL, BENCHMARK_NONE }
        num_trials: number of trials to run
        num_results: the number of results that will come out of the model function. If num_results is
            greater than 1, it should be indexed by range(num_results)
        is_wandb: whether wandb experiment tracking is enabled
        wandb_project: the string name of the wandb project
        wandb_entity: the wandb user
        args: args to add the the wandb config

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

            rand_key, subkey = random.split(rand_key)
            data = get_data(subkey, **data_kwargs)

            for k, (model_name, model, model_kwargs) in enumerate(models):
                print(f"trial {i} {benchmark}: {benchmark_val} {model_name}")
                name = f"{model_name}_{benchmark}{benchmark_val}_t{i}"

                if benchmark_type == BENCHMARK_MODEL:
                    model_kwargs = {**model_kwargs, benchmark: benchmark_val}

                if is_wandb:
                    wandb.init(
                        project=wandb_project,
                        entity=wandb_entity,
                        name=name,
                        settings=wandb.Settings(start_method="fork"),
                    )
                    wandb.config.update(args)
                    type_list = [str, int, float, bool]

                    def display_val(val):
                        if isinstance(val, enum.Enum):
                            return val.name
                        elif type(val) in type_list or val is None:
                            return val
                        else:
                            return type(val)

                    wandb.config.update(
                        {key: display_val(val) for key, val in model_kwargs.items()},
                        allow_val_change=True,
                    )
                    wandb.config.update({"model_name": model_name})

                rand_key, subkey = random.split(rand_key)
                res = model(data, subkey, name, **model_kwargs)

                if is_wandb:
                    wandb.finish()

                if num_results > 1:
                    for q in range(num_results):
                        results[i, j, k, q] = res[q]
                else:
                    results[i, j, k, 0] = res

    return results


def benchmark_lr(
    get_data: Callable,
    models: list[tuple[str, Callable, dict]],
    rand_key: ArrayLike,
    lr_range: Sequence[float],
    num_trials: int = 1,
    num_results: int = 1,
    is_wandb: bool = False,
    wandb_project: str = "",
    wandb_entity: str = "",
    args: dict = {},
) -> np.ndarray:
    """
    The most common usecase of the benchmark function is benchmarking over a learning rate range.
    If the lr_range has no values, instead this defaults to no benchmarking, just over the model
    list and number of trials.

    args:
        get_data: function that takes as its first argument the benchmark_value, and a rand_key
            as its second argument. It returns the data which later gets passed to model.
        models: the elements of the tuple are (str) model_name, (func) model, and a dict of keyword
            arguments to pass to model. Model is a function that takes data, a rand_key, the
            model_name, and remaining keyword arguments and returns either a single float score
            or an iterable of length num_results of float scores.
        rand_key: key for randomness
        num_trials: number of trials to run
        num_results: the number of results that will come out of the model function. If num_results is
            greater than 1, it should be indexed by range(num_results)
        is_wandb: whether wandb experiment tracking is enabled
        wandb_project: the string name of the wandb project
        wandb_entity: the wandb user
        args: args to add the the wandb config

    returns:
        an np.array of shape (trials, lr_range, models, num_results) with the results all filled in
    """
    benchmark_type = BENCHMARK_MODEL if len(lr_range) else BENCHMARK_NONE
    return benchmark(
        get_data,
        models,
        rand_key,
        "lr",
        lr_range,
        benchmark_type,
        num_trials,
        num_results,
        is_wandb,
        wandb_project,
        wandb_entity,
        args,
    )


class Mapper:
    """
    Functor for map_and_loss in train, map_loss_in_batches, etc, where arguments can be provided
    beforehand. In this case, it is useful for smse vs relative error, and whether to learn the
    residual or not.
    """

    losses: list[geom.Losses]
    residual: bool
    reduce: str | None
    eps: float

    def __init__(
        self: Self,
        losses: list[geom.Losses],
        residual: bool = False,
        reduce: str | None = "mean",
        eps: float = 0,
    ) -> None:
        """
        Docstring for __init__

        args:
            losses: a list of losses, must be at least 1
            residual: Whether the network should learn the residual, defaults to False
            reduce: How to reduce the batch dimension, defaults to 'mean' but can also be None
            eps: epsilon value to use for nrmse and lr_rel, avoid dividing by 0
        """
        assert len(losses) > 0, "Mapper::init: At least one loss required."
        self.losses = losses
        self.residual = residual
        self.reduce = reduce
        self.eps = eps

    @eqx.filter_jit
    def map(
        self: Self,
        model: models.MultiImageModule,
        multi_image_x: geom.MultiImage,
        aux_data: eqx.nn.State | None = None,
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        """
        The map function using the model and the input data.
        """
        out, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
            multi_image_x, aux_data
        )

        if self.residual:
            # add the last timestep to the residual
            pred_y = out.empty()
            for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), out.values()):
                pred_y.append(k, parity, img_in[:, -1:] + img_resid)

            return pred_y, aux_data
        else:
            return out, aux_data

    @eqx.filter_jit
    def __call__(
        self: Self,
        model: models.MultiImageModule,
        multi_image_x: geom.MultiImage,
        multi_image_y: geom.MultiImage,
        aux_data: eqx.nn.State | None = None,
    ) -> tuple[jax.Array, eqx.nn.State | None]:
        """
        Equivalent of the map_and_loss function.
        """
        pred_y, aux_data = self.map(model, multi_image_x, aux_data)

        loss_outputs = []
        for loss in self.losses:  # the order is important
            if loss is geom.Losses.SMSE:
                loss_outputs.append(smse_loss(pred_y, multi_image_y, self.reduce))
            elif loss is geom.Losses.NRMSE:
                loss_outputs.append(nrmse_loss(pred_y, multi_image_y, self.reduce, eps=self.eps))
            elif loss is geom.Losses.NRMSE_PER_PIXEL:
                loss_outputs.append(
                    nrmse_per_pixel_loss(pred_y, multi_image_y, self.reduce, eps=self.eps)
                )
            elif loss is geom.Losses.L2_REL:
                loss_outputs.append(l2_rel_error(pred_y, multi_image_y, self.reduce, eps=self.eps))
            elif loss is geom.Losses.L2_REL_PER_PIXEL:
                loss_outputs.append(
                    l2_per_pixel_rel_error(pred_y, multi_image_y, self.reduce, eps=self.eps)
                )

        # if we aren't reducing the batch dimension, we don't want to squeeze it out
        loss_outputs = jnp.stack(loss_outputs, axis=-1)
        if len(self.losses) == 1:
            squeeze_outputs = jnp.squeeze(loss_outputs, axis=1 if self.reduce is None else 0)
        else:
            squeeze_outputs = loss_outputs

        return squeeze_outputs, aux_data
