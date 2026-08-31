import argparse
import h5py
import numpy as np
import pathlib
import time
from typing_extensions import Self

import jax.numpy as jnp
import jax
import jax.random as random
from jaxtyping import PRNGKeyArray
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
import ginjax.utils as utils
import ginjax.data as gc_data
import ginjax.models as models
import ginjax.ml as ml
from . import anyd_helpers


def read_one_h5(
    D: int, is_torus: bool, past_steps: int, data_dict: h5py.File, traj_idxs: np.ndarray | slice
) -> tuple[geom.MultiImage, geom.MultiImage]:
    """
    Given a dimension and filename, read the data and return as jax arrays.

    args:
        D: the dimension of the space
        filename: the full file path
        num_trajectories: number of trajectories to read

    returns:
        density, pressure, and velocity fields
    """
    density = jax.device_put(jnp.array(data_dict["density"][traj_idxs][()]), jax.devices("cpu")[0])

    pressure = jax.device_put(
        jnp.array(data_dict["pressure"][traj_idxs][()]), jax.devices("cpu")[0]
    )
    if len(density) == 0:
        density_pressure = jnp.zeros((0, 2 * density.shape[1]) + density.shape[2:])
    else:
        # (batch,2,timesteps,spatial)
        density_pressure = jnp.concatenate([density[:, None], pressure[:, None]], axis=1)
        # (batch,2*timesteps,spatial)
        density_pressure = density_pressure.reshape(
            (len(density_pressure), -1) + density_pressure.shape[3:]
        )

    velocities = []
    for vkey in ["Vx", "Vy", "Vz"][:D]:
        velocities.append(
            jax.device_put(jnp.array(data_dict[vkey][traj_idxs][()]), jax.devices("cpu")[0])
        )

    velocity = jnp.stack(velocities, axis=-1)

    constant_fields = geom.MultiImage({}, D, is_torus)

    X, Y = gc_data.batch_time_series(
        geom.MultiImage({(0, 0): density_pressure, (1, 0): velocity}, D, is_torus),
        constant_fields,
        velocity.shape[1],
        past_steps,
        1,
    )

    return X, Y


def open_hdf5(D: int, data_dir: str) -> h5py.File:
    data_dir_path = pathlib.Path(data_dir)
    fnames = {
        2: "2D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_64_Train.hdf5",
        3: "3D_CFD_Rand_M0.1_Eta1e-08_Zeta1e-08_periodic_64_Train.hdf5",
    }
    return h5py.File(data_dir_path / f"cfd_{D}d" / fnames[D])


class CFDDataset(ml.MultiImageDataset):
    """
    A datset to load the cfd data. This dataset loads all the data into memory, then takes a subset
    during __call__. The only difference between this and ml.MultiImageDataset is how it handles
    the CFD data specifically in the constructor.

    The __getitem__ for this class expects a list of integer indices for the entire batch at
    once, which means the sampler of the data loader should be a batch sampler.
    """

    def __init__(
        self: Self,
        D: int,
        n_data: int,
        idx_shift: int,
        past_steps: int,
        data_dir: str,
        devices: list[jax.Device] | None = None,
        use_devices: bool = True,
        is_torus: bool = True,
    ) -> None:
        self.D = D

        data_dict = open_hdf5(D, data_dir)
        X, Y = read_one_h5(D, is_torus, past_steps, data_dict, slice(idx_shift, idx_shift + n_data))
        data_dict.close()

        self.X = X
        self.Y = Y

        self.devices = devices if devices else jax.devices()
        self.use_devices = use_devices


class CFDDatasetLazy(ml.MultiImageDataset):
    """
    A dataset to lazily load cfd data.

    The __getitem__ for this class expects a list of integer indices for the entire batch at
    once, which means the sampler of the data loader should be a batch sampler.
    """

    D: int
    n_data: int
    idx_shift: int
    past_steps: int
    samples_per_traj: int
    data_dict: h5py.File
    devices: list[jax.Device]
    use_devices: bool
    is_torus: bool

    def __init__(
        self: Self,
        D: int,
        n_data: int,
        idx_shift: int,
        past_steps: int,
        data_dir: str,
        devices: list[jax.Device] | None = None,
        use_devices: bool = True,
        is_torus: bool = True,
    ) -> None:
        self.D = D
        self.n_data = n_data
        self.idx_shift = idx_shift
        self.past_steps = past_steps
        self.samples_per_traj = 21 - args.past_steps

        self.data_dict = open_hdf5(D, data_dir)

        self.devices = devices if devices else jax.devices()
        self.use_devices = use_devices
        self.is_torus = is_torus

    def __len__(self: Self) -> int:
        return self.n_data * self.samples_per_traj

    def __getitem__(self: Self, idx: list[int]) -> tuple[geom.MultiImage, geom.MultiImage]:
        # all of these are shape (num_trajectories, timesteps, spatial, tensor)
        # 1D: (10K,101,1024,tensor)
        # 2D: (1K,21,512,512,tensor) or (10K,21,128,128,tensor)
        # 2D: my janky homebrew is (9990,21,128,128,tensor)
        # 3D: (100,21,128,128,128,tensor)

        # TODO: may be a way to do this better with a memory mask

        idxs = np.sort(np.array(idx))
        shifted_idxs = idxs + self.idx_shift * self.samples_per_traj

        # select only the trajectories that we need
        # get the idxs and counts of the trajectories, e.g. [14,15,35,171] -> [0,2,11], [2,1,1]
        traj_idxs, idxs_counts = np.unique(
            shifted_idxs // self.samples_per_traj, return_counts=True
        )

        X, Y = read_one_h5(self.D, self.is_torus, self.past_steps, self.data_dict, traj_idxs)

        # get the row numbers of loaded data, e.g. [14,15,35,171] -> [0,0,1,2]
        row_numbers = np.repeat(np.arange(len(traj_idxs)), idxs_counts)
        # get the corrected full indices [14,15,35,171] % 17 + [0,0,1,2]*17
        samples_idxs = idxs % self.samples_per_traj + row_numbers * self.samples_per_traj
        samples_idxs = jnp.array(samples_idxs)

        X, Y = X.get_subset(samples_idxs), Y.get_subset(samples_idxs)

        if self.use_devices:
            X, Y = X.reshape_pmap(self.devices), Y.reshape_pmap(self.devices)

        return X, Y

    def get_N(self: Self) -> int:
        return self.data_dict["density"].shape[2]


def get_data(
    D: int,
    n_train: int,
    n_val: int,
    n_test: int,
    past_steps: int,
    batch_size: int,
    data_dir: str,
) -> tuple[DataLoader[CFDDataset], DataLoader[CFDDataset], DataLoader[CFDDataset], float]:
    """
    TODO: maybe I can load data as needed? one batch at a time? probably needed for 3d data
    Get data of a particular dimension from a preset list of files in the specified folder.

    args:
        D: the dimension of the data to get
        n_train: number of training trajectories
        n_val: number of validation trajectories
        n_test: number of test trajectories
        past_steps: number of historical steps to use to predict the next 1 step
        batch_size: the batch size
        data_dir: the direcory where the data is stored

    returns:
        input and output pairs of multi images for train, val, and test, plus data generation time
            for train
    """
    if D == 2:
        train_dataset = CFDDataset(D, n_train, 0, past_steps, data_dir)
        val_dataset = CFDDataset(D, n_val, n_train, past_steps, data_dir)
        test_dataset = CFDDataset(D, n_test, n_train + n_test, past_steps, data_dir)
    elif D == 3:
        train_dataset = CFDDatasetLazy(D, n_train, 0, past_steps, data_dir)
        val_dataset = CFDDatasetLazy(D, n_val, n_train, past_steps, data_dir)
        test_dataset = CFDDatasetLazy(D, n_test, n_train + n_test, past_steps, data_dir)
    else:
        raise ValueError(f"cfd_anyd::get_data: expects D=2,3, but got D={D}")

    # RandomSampler requires __len__ > 0
    sampler = RandomSampler(train_dataset) if n_train > 0 else SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=BatchSampler(sampler, batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )
    val_dataloader = DataLoader(
        val_dataset,
        sampler=BatchSampler(SequentialSampler(val_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )
    test_dataloader = DataLoader(
        test_dataset,
        sampler=BatchSampler(SequentialSampler(test_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    data_generation_time = 0.0  # TODO: fix this

    return train_dataloader, val_dataloader, test_dataloader, data_generation_time


# possible run
# CUDA_VISIBLE_DEVICES=0,1 time python3 -m scripts.cfd_anyd --data /data/wgregor4/pdebench/
# --n-train 128 --n-val 32 --n-test 128 --model-dir /data/wgregor4/runs/cfd_anyd/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32",
    )
    parser.add_argument(
        "--rescale-list",
        help="the types of rescalings to do, defaults to spin_embed, for ablations do spin_embed,copy,zeros",
        type=lambda s: s.split(","),
        default="spin_embed",
    )
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=4
    )
    parser.add_argument("--batch-train", help="batch size for 2D training", type=int, default=32)
    # you can do 4 (2 per gpu) with equiv48, but not 8
    parser.add_argument("--batch-tune", help="batch size for 3D tuning", type=int, default=4)
    parser.add_argument(
        "--pretrain-lr-range",
        help="benchmark trained model over the lr, turns on wandb",
        type=lambda s: tuple(float(x) for x in s.split(",")) if isinstance(s, str) else None,
        default=None,
    )
    parser.add_argument(
        "--finetune-lr-range",
        help="benchmark tuned model over the lr, turns on wandb",
        type=lambda s: tuple(float(x) for x in s.split(",")) if isinstance(s, str) else None,
        default=None,
    )
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="cfd-anyd")

    return parser.parse_args()


# MAIN
args = handleArgs()
if args.wandb:
    print("Use --train-wandb or --test-wandb to control these individually, instead of --wandb.")
    exit()

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

rescale_options = {
    "spin_embed": geom.Rescaling.SPIN_EMBED,
    "copy": geom.Rescaling.COPY,
    "zeros": geom.Rescaling.ZEROS,
}
rescale_list = [rescale_options[x.lower()] for x in args.rescale_list]

pretrain_lr = args.pretrain_lr_range is not None
finetune_lr = args.finetune_lr_range is not None

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

# D=1 doesn't make sense for a vector field, so we restrict the problem to only this case
train_D = 2
test_D = 3
full_D_range = (train_D, test_D)
free_filters_dict, upsample_filters_dict = anyd_helpers.generate_filters(full_D_range, [0, 1, 2])

train_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "batch_size": args.batch_train,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": pretrain_lr,
    "wandb_project": args.wandb_project,
    "wandb_entity": args.wandb_entity,
}

baseline_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "batch_size": args.batch_tune,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": args.images_dir,  # currently ignored
    "verbose": args.verbose,
    "is_wandb": finetune_lr,
    "wandb_project": args.wandb_project,
    "wandb_entity": args.wandb_entity,
}

test_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "batch_size": args.batch_tune,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": finetune_lr,
    "wandb_project": args.wandb_project,
    "wandb_entity": args.wandb_entity,
}

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt = CFDDataset(D, 1, 0, args.past_steps, args.data, use_devices=False)[[0]]
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    model_list = []
    for trial in range(args.n_trials):
        key, *subkeys = random.split(key, num=10)
        model_list.extend(
            [
                (
                    f"unetBase_equiv48_trial{trial}",
                    models.UNet(
                        D,
                        input_keys,
                        output_keys,
                        depth=48,
                        activation_f=jax.nn.gelu,
                        conv_filters=free_filters_dict[D],
                        upsample_filters=upsample_filters_dict[D],
                        key=subkeys[2],
                    ),
                    {  # train_kwargs
                        "lr": {2: {128: 1e-4}, 3: {0: 5e-4, 1: 5e-4, 4: 1e-4, 32: 1e-4}},
                        # D=3, for all of them (and tuning) its just 1e-4
                        **train_kwargs,
                    },
                    {  # train_kwargs
                        "lr": {2: {128: 1e-4}, 3: {0: 5e-4, 1: 5e-4, 4: 1e-4, 32: 1e-4}},
                        **baseline_kwargs,
                    },
                    {  # tune and eval kwargs
                        "lr": {2: {3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 32: 1e-4}}},
                        "conv_filters_dict": free_filters_dict,
                        "upsample_filters_dict": upsample_filters_dict,
                        **test_kwargs,
                    },
                ),
            ]
        )

    model_list_d[D] = model_list


# extended lambda function
def get_data_lambda(
    D: int, n_train: int, n_val: int, n_test: int, batch_size: int, key: PRNGKeyArray
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    float,
]:
    return get_data(D, n_train, n_val, n_test, args.past_steps, batch_size, args.data)


key, subkey = random.split(key)
anyd_helpers.run_anyd(
    (train_D,),
    (test_D,),
    args.n_tune_range,
    args,
    subkey,
    get_data_lambda,
    model_list_d,
    args.pretrain_lr_range,
    args.finetune_lr_range,
    rescale_list,
    {2: args.batch_train, 3: args.batch_tune},
)
