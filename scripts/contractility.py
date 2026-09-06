import argparse
import math
import numpy as np
import os
import pathlib
import time
from typing import Literal

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Float, PRNGKeyArray
import optax
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils
from ginjax.data import batch_time_series


def read_one(
    fname: pathlib.Path,
) -> tuple[
    Float[Array, " spatial"],
    Float[Array, " spatial"],
    Float[Array, "spatial D"],
    Float[Array, " spatial"],
]:
    # shape (channels,spatial)
    data = jnp.array(np.load(fname), device=jax.devices("cpu")[0])
    zyxin = data[6]  # shape (spatial,)
    actin = data[7]

    fx = data[2]
    fy = data[3]
    force = jnp.stack([fx, fy], axis=-1)  # (spatial,tensor)

    # make is (spatial,) of 0 for outside cell, 255 for inside cell. Convert to 0 and 1.
    mask = (data[4] != 0).astype(int)

    # compare mask and force mask
    print(
        "mean force inside",
        jnp.mean(
            jnp.linalg.norm(jnp.stack([fx[data[4] != 0], fy[data[4] != 0]], axis=-1), axis=-1)
        ),
    )
    print(
        "mean force outside",
        jnp.mean(
            jnp.linalg.norm(jnp.stack([fx[data[4] == 0], fy[data[4] == 0]], axis=-1), axis=-1)
        ),
    )

    print(jnp.sum(jnp.abs(data[4] - data[5])))
    exit()

    return zyxin, actin, force, mask


def read_cell(
    cell_dir: pathlib.Path,
) -> tuple[
    Float[Array, "steps spatial"],
    Float[Array, "steps spatial"],
    Float[Array, "steps spatial D"],
    Float[Array, "steps spatial"],
]:
    frame_files = os.listdir(cell_dir)
    # sort by the frame number, files are "yadayada_<frame>.npy".
    sorted_frames = sorted(frame_files, key=lambda s: int(s[s.rfind("_") + 1 : s.rfind(".npy")]))
    zyxin_ls = []
    actin_ls = []
    force_ls = []
    mask_ls = []
    for frame_file in sorted_frames:  # these need to be sorted properly
        zyxin, actin, force, mask = read_one(cell_dir / frame_file)
        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)
        mask_ls.append(mask)

        if len(mask_ls) > 1:
            latest_mask = mask_ls[-1]
            prev_mask = mask_ls[-2]
            print("expanded", jnp.sum((latest_mask - prev_mask) > 0))
            print("contracted", jnp.sum((prev_mask - latest_mask) > 0))

    exit()

    return jnp.stack(zyxin_ls), jnp.stack(actin_ls), jnp.stack(force_ls), jnp.stack(mask_ls)


def read_cells(
    D: int, cell_dirs: list[pathlib.Path], normalize: Literal["previous", "mean_std"]
) -> geom.MultiImage:
    """
    Read the cells and create a multi image out of them.

    args:
        D: the dimension of the space
        cell_dirs: a list of the cell directories where all the frames live

    returns:
        a multi image with shape (batch,channel*time,spatial,tensor)
    """
    zyxin_ls = []
    actin_ls = []
    force_ls = []
    for cell_dir in cell_dirs:  # do they all have the same number of timesteps?
        zyxin, actin, force, mask = read_cell(cell_dir)

        if normalize == "previous":
            # follow the normalization done in the previous paper
            zyxin = (zyxin - jnp.mean(zyxin[mask == 0])) / (
                jnp.mean(zyxin[mask != 0]) - jnp.mean(zyxin[mask == 0])
            )
            actin = (actin - jnp.mean(actin[mask == 0])) / (
                jnp.mean(actin[mask != 0]) - jnp.mean(actin[mask == 0])
            )
            force_norm = jnp.linalg.norm(force, axis=-1)  # (steps,spatial)
            # TODO: do I need to use the mask here?
            force = force / jnp.mean(force_norm)
        elif normalize == "mean_std":
            zyxin = (zyxin - jnp.mean(zyxin)) / jnp.std(zyxin)
            actin = (actin - jnp.mean(actin)) / jnp.std(actin)
            force = force / jnp.std(jnp.linalg.norm(force, axis=-1))

        # print(jnp.mean(zyxin), jnp.std(zyxin))
        # print(jnp.mean(actin), jnp.std(actin))
        # print(
        #     jnp.mean(jnp.linalg.norm(force, axis=-1)),
        #     jnp.std(jnp.linalg.norm(force, axis=-1)),
        # )
        zyxin_ls.append(zyxin)
        actin_ls.append(actin)
        force_ls.append(force)

    # (batch,time,spatial,tensor)
    zyxins = jnp.stack(zyxin_ls)
    actins = jnp.stack(actin_ls)
    forces = jnp.stack(force_ls)
    # TODO: does mask ever grow, or does the cell only contract and get smaller?

    # (batch,channel,time,spatial,tensor) -> (batch,channel*time,spatial,tensor)
    scalars = jnp.stack([zyxins, actins], axis=1).reshape(len(cell_dirs), -1, *zyxins.shape[2:])
    return geom.MultiImage({((), 0): scalars, ((False,), 0): forces}, D, is_torus=False)


def get_data(
    D: int,
    data_dir: pathlib.Path,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    future_steps: int,
    batch_size: int,
    normalize: Literal["previous", "mean_std"],
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    geom.Signature,
    geom.Signature,
]:
    """
    Load the data and put it into multi image datasets.
    """
    # TODO: Probably going to want some kind of normalization

    # use cell_1 as the test, as in the repo?

    cell_dirs = [
        data_dir / x
        for x in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, x)) and ("cell" in x)
    ]
    test_cells = cell_dirs[-1:]  # [cell_1]
    cells = cell_dirs[:-1]  # [cell_0, cell_2, cell_3]

    train_val = read_cells(D, cells, normalize)
    test = read_cells(D, test_cells, normalize)
    total_timesteps = train_val[((False,), 0)].shape[1]

    train_val_x, train_val_y = batch_time_series(
        train_val, geom.MultiImage({}, D, False), total_timesteps, past_steps, future_steps
    )

    train_x = train_val_x.get_subset(jnp.arange(n_train))
    train_y = train_val_y.get_subset(jnp.arange(n_train))
    val_x = train_val_x.get_subset(jnp.arange(n_train, n_train + n_val))
    val_y = train_val_y.get_subset(jnp.arange(n_train, n_train + n_val))

    test_x, test_y = batch_time_series(
        test, geom.MultiImage({}, D, False), total_timesteps, past_steps, future_steps
    )

    train_dataset = ml.MultiImageDataset(train_x, train_y)
    val_dataset = ml.MultiImageDataset(val_x, val_y)
    test_dataset = ml.MultiImageDataset(test_x, test_y)

    train_dataloader = DataLoader(
        train_dataset,
        sampler=BatchSampler(
            RandomSampler(train_dataset), batch_size, drop_last=n_train > batch_size
        ),
        collate_fn=lambda x: x[0],
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=BatchSampler(
            SequentialSampler(val_dataset), batch_size, drop_last=n_val > batch_size
        ),
        collate_fn=lambda x: x[0],
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=BatchSampler(
            SequentialSampler(test_dataset), batch_size, drop_last=n_test > batch_size
        ),
        collate_fn=lambda x: x[0],
    )

    return (
        train_dataloader,
        val_dataloader,
        test_dataloader,
        train_x.get_signature(),
        train_y.get_signature(),
    )


def train_and_eval(
    data: tuple[
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
        DataLoader[ml.MultiImageDataset],
    ],
    key: PRNGKeyArray,
    model_name: str,
    model: models.MultiImageModule,
    lr: float,
    batch_size: int,
    epochs: int,
    rollout_steps: int,
    model_dir: pathlib.Path | None,
    overwrite_save_model: bool,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[Float[Array, ""], ...]:
    train_dl, val_dl, test_dl = data
    assert isinstance(train_dl.dataset, ml.MultiImageDataset)
    batch_stats = eqx.nn.State(model) if has_aux else None

    print(f"Model params: {models.count_params(model):,}")

    mapper = ml.Mapper([geom.Losses.NRMSE], eps=1e-5)

    model_path = model_dir / f"{model_name}.eqx" if model_dir else None
    if model_path and model_path.is_file() and not overwrite_save_model:
        trained_model, _ = ml.load_plus(model_path, model)
    else:
        steps_per_epoch = int(math.ceil(len(train_dl.dataset) / batch_size))
        trained_model, _, _, _, train_time = ml.train_dl(
            train_dl,
            mapper,
            model,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            val_dataloader=val_dl,
            val_map_and_loss=ml.Mapper([geom.Losses.NRMSE], eps=1e-5),
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if model_path:
            assert not model_path.is_file() or overwrite_save_model
            # TODO: need to save batch_stats as well
            ml.save_plus(model_path, trained_model, {"train_time": train_time})

    train_loss = ml.map_loss_in_batches_dl(mapper, model, train_dl)
    val_loss = ml.map_loss_in_batches_dl(mapper, model, val_dl)
    test_loss = ml.map_loss_in_batches_dl(mapper, model, test_dl)

    print(f"Test Loss: {test_loss}")
    # rollout_mapper = ml.Mapper()
    # TODO: rollout loss

    return train_loss, val_loss, test_loss


def handleArgs() -> argparse.Namespace:
    """
    CUDA_VISIBLE_DEVICES=3 time python3 -m scripts.contractility \
    --data /data/wgregor4/contractility/ZyxAct_16kPa_small/ \
    --n-train 256 --n-val 128 --n-test 8 -b 2
    """
    parser = utils.get_common_parser()
    parser.add_argument(
        "--past-steps", help="the number of past steps for the input", type=int, default=4
    )
    parser.add_argument(
        "--future-steps", help="the number of future steps to output", type=int, default=1
    )
    parser.add_argument(
        "--rollout-steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--normalize-type",
        help="type of normalization, `previous` of the method from the earlier paper",
        choices=["previous", "mean_std"],
        default="previous",
    )
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="contractility"
    )

    return parser.parse_args()


# MAIN
args = handleArgs()

# Since we only have 4 trajectories, n_train and n_val refer to the number of data points after
# reshaping timesteps into batches.
D = 2
data_dir = pathlib.Path(args.data)

train_dl, val_dl, test_dl, input_keys, output_keys = get_data(
    D,
    data_dir,
    args.n_train,
    args.n_val,
    args.n_test,
    args.past_steps,
    args.future_steps,
    args.batch,
    args.normalize_type,
)

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()


key = jax.random.PRNGKey(time.time_ns()) if (args.seed is None) else jax.random.PRNGKey(args.seed)

group_actions = geom.make_all_operators(D)
upsample_filters = geom.get_invariant_filters(
    Ms=[2], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)
conv_filters = geom.get_invariant_filters(
    Ms=[3], ks=[0, 1, 2], parities=[0], D=D, operators=group_actions
)

train_kwargs = {
    "batch_size": args.batch,
    "epochs": args.epochs,
    "rollout_steps": args.rollout_steps,
    "model_dir": args.model_dir,
    "overwrite_save_model": args.overwrite_save_model,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = jax.random.split(key, num=13)
model_list = [
    # (
    #     # batch=2 works, batch=4 fails
    #     "unetBase_equiv20",
    #     train_and_eval,
    #     {
    #         "model": models.UNet(
    #             D,
    #             input_keys,
    #             output_keys,
    #             depth=20,
    #             activation_f=jax.nn.gelu,
    #             conv_filters=conv_filters,
    #             upsample_filters=upsample_filters,
    #             key=subkeys[8],
    #         ),
    #         "lr": 4e-4,  # 4e-4 to 6e-4 works, larger sometimes explodes
    #         **train_kwargs,
    #     },
    # ),
    (
        "unetBase",
        train_and_eval,
        {
            "model": models.UNet(
                D,
                input_keys,
                output_keys,
                depth=64,
                use_bias=True,
                activation_f=jax.nn.gelu,
                equivariant=False,
                kernel_size=3,
                use_group_norm=False,
                padding_mode="ZEROS",
                key=subkeys[6],
            ),
            "lr": 8e-4,
            **train_kwargs,
        },
    ),
]

key, subkey = jax.random.split(key)

# Use this for benchmarking the models with known learning rates.
results = ml.benchmark_lr(
    lambda _: (train_dl, val_dl, test_dl),
    model_list,
    subkey,
    [],  # lr_range
    num_trials=args.n_trials,
    num_results=3 + args.rollout_steps,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
    # args=args, # needs to be a dictionary
)
