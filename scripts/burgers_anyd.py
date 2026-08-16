import argparse
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np
import pathlib
import time

import jax
import jax.numpy as jnp
from jax import random
from jaxtyping import PRNGKeyArray
import apebench
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
from ginjax import models
from ginjax import ml
import ginjax.data as gc_data
from ginjax import utils
from . import anyd_helpers


def get_data_d(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    n_timesteps: int,
    subsample: int,  # new
    n_batch: int,
    key: jax.Array,
    data_dir: pathlib.Path,
) -> tuple[geom.MultiImage, geom.MultiImage, float]:
    """
    Generate data for a particular dimension using the apebench code.

    args:
        D: dimension
        N: side length of images
        diffusion_coef: parameter for burgers
        convection_coef: parameter for burgers
        subsample: additional steps to generate, which are then subsampled for the output
        n_batch: batch size, or total dataset size in this case
        key: jax key for randomness
        data_dir: director to save/load the data

    returns:
        a tuple of the input and output geometric images
    """
    is_torus = True
    n_timesteps_int = n_timesteps * subsample  # integrator time steps
    n_warmup_steps = 0  # in 3D, seems like there is an initial problem
    scenario = "diff_burgers"  # diff setting guaranteed to avoid NaNs, so prefer it over norm, phy

    # we multiply the coefs by D to remove the dimension normalizing effect from diff_burgers
    scaled_diff_coef = diffusion_coef * D
    scaled_conv_coef = convection_coef * D

    # information about relationship:
    # https://github.com/Ceyron/exponax/blob/main/exponax/stepper/generic/_convection.py
    # default values are given (diffusion_gamma, convection_delta)
    # apebench.scenarios.physical.Burgers()  # (0.0003,-0.125)
    # apebench.scenarios.normalized.Burgers()  # (0.00003,-0.0125)
    # apebench.scenarios.difficulty.Burgers()  # (1.5,-1.5)

    train_name = f"D{D}_{scenario}_N{N}_n{n_batch}_diffusion{scaled_diff_coef}_convection{scaled_conv_coef}_t{n_timesteps_int}"
    train_path = pathlib.Path(f"{data_dir}") / f"{train_name}_train.npy"
    if not train_path.is_file():
        start_time = time.time()
        print(f"Generating data:", train_path)
        key, subkey = random.split(key)
        train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

        apebench.scraper.scrape_data_and_metadata(
            data_dir,  # warning, but it works
            scenario=scenario,
            name=train_name,
            num_spatial_dims=D,
            num_points=N,
            num_warmup_steps=n_warmup_steps,
            num_train_samples=n_batch,
            num_test_samples=0,
            train_seed=int(train_seed),
            test_seed=int(test_seed),
            train_temporal_horizon=n_timesteps_int - 1,
            test_temporal_horizon=n_timesteps_int - 1,
            # diffusion_coef=diffusion_coef,  # for phy_burgers
            # convection_coef=convection_delta,  # for phy_burgers
            # diffusion_alpha=diffusion_coef,  # for norm_burgers
            # convection_beta=convection_coef,  # for norm_burgers
            diffusion_gamma=scaled_diff_coef,  # for diff_burgers
            convection_delta=scaled_conv_coef,  # for diff_burgers
        )

        data_generation_time = time.time() - start_time
        print(f"Finished: {data_generation_time} seconds.")
    else:
        # dictionary by D, then n_batch
        # Since these values aren't saved with the data, hardcode them from some previous run.
        data_generation_times = {
            64: {
                2: {
                    0: jnp.array([1.0652430057525635, 0.40758848190307617]),
                    8: jnp.array([7.801406621932983, 3.1244280338287354]),
                },
                3: {
                    0: jnp.array([0.6283245086669922, 2.0740256309509277, 0.9826910495758057]),
                    1: jnp.array([5.805211067199707]),
                    4: jnp.array([7.013747453689575]),
                    8: jnp.array([8.912535429000854, 5.779284477233887, 7.701824903488159]),
                },
            },
            96: {  # 96 was also done with n_timesteps=25, rather than 50
                2: {
                    0: jnp.array([0.6557881832122803, 0.3366715908050537]),
                    8: jnp.array([4.890353679656982, 2.4403231143951416]),
                },
                3: {
                    0: jnp.array([0.3930032253265381, 0.38234496116638184]),
                    1: jnp.array([0]),
                    4: jnp.array([0]),
                    8: jnp.array([15.993224382400513]),
                },
            },
        }
        data_generation_time = float(jnp.mean(data_generation_times[N][D][n_batch]))

    cpu = jax.devices("cpu")[0]
    # (batch,timesteps,tensor,spatial) -> (batch,timesteps,spatial,tensor)
    train_data = jnp.moveaxis(jax.device_put(jnp.load(train_path)[:, ::subsample], cpu), 2, -1)
    # subsample here for memory efficiency

    # Plot the training data
    # print(train_data.shape)
    # vmax = float(jnp.max(jnp.abs(train_data)))
    # vmin = -1 * vmax

    # print(vmax)

    # timesteps = 10

    # nrows = D
    # ncols = timesteps
    # # figsize is 6 per col, 6 per row, (cols,rows)
    # fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    # for component in range(D):
    #     comp_name = ["x", "y", "z"][component]

    #     print(int(n_timesteps / timesteps))
    #     timestep_images = train_data[0, :: int(n_timesteps / timesteps), ..., component]
    #     print(timestep_images.shape)
    #     for i, timestep_image in enumerate(timestep_images):
    #         geom.GeometricImage(timestep_image, 0, D).plot(
    #             axes[component, i],
    #             title=f"time {i} {comp_name}",
    #             vmin=vmin,
    #             vmax=vmax,
    #             colorbar=True,
    #         )

    # plt.tight_layout()
    # plt.savefig("/data/wgregor4/images/apebench/burgers/spaced_10_steps.png")
    # plt.close(fig)

    # exit()

    constant_fields = geom.MultiImage({}, D, is_torus)
    x0, xt = gc_data.batch_time_series(
        geom.MultiImage({(1, 0): train_data}, D, is_torus), constant_fields, n_timesteps, 1, 1
    )

    return x0, xt, data_generation_time


def get_data(
    D: int,
    N: int,
    diffusion_coef: float,
    convection_coef: float,
    n_timesteps: int,
    subsample: int,  # new
    n_train: int,
    n_val: int,
    n_test: int,
    batch_size: int,
    key: jax.Array,
    data_dir: str,
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    float,
]:
    """
    Generate full dataset with train, validation, and test data sets.

    args:
        D: dimension
        N: side length of images
        diffusion_coef: parameter for burgers
        convection_coef: parameter for burgers
        subsample: additional steps to generate, which are then subsampled for the output
        n_train: train dataset size
        n_val: validation dataset size
        n_test: test dataset size
        batch_size: the training batch size
        key: jax key for randomness
        data_dir: director to save/load the data

    returns:
        tuple of geometric images, (train_X, train_Y, val_X, val_Y, test_X, test_Y)
    """
    data_dir_path = pathlib.Path(data_dir)

    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_x0, train_xt, train_data_time = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_train,
        subkey1,
        data_dir_path / "train",
    )
    train_dataset = ml.MultiImageDataset(train_x0, train_xt)
    # RandomSampler breaks if the dataset is empty
    sampler = RandomSampler(train_dataset) if n_train > 0 else SequentialSampler(train_dataset)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=BatchSampler(sampler, batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    val_x0, val_xt, _ = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_val,
        subkey2,
        data_dir_path / "val",
    )
    val_dataset = ml.MultiImageDataset(val_x0, val_xt)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=BatchSampler(SequentialSampler(val_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    test_x0, test_xt, _ = get_data_d(
        D,
        N,
        diffusion_coef,
        convection_coef,
        n_timesteps,
        subsample,
        n_test,
        subkey3,
        data_dir_path / "test",
    )
    test_dataset = ml.MultiImageDataset(test_x0, test_xt)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=BatchSampler(SequentialSampler(test_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    return train_dataloader, val_dataloader, test_dataloader, train_data_time


# Something like
# CUDA_VISIBLE_DEVICES=0,1 time python3 -m scripts.burgers_anyd --data /data/wgregor4/apebench/burgers/
# --n-train 8 --n-val 8 --n-test 8 --model-dir /data/wgregor4/runs/burgers_anyd/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,8",
    )
    parser.add_argument(
        "--rescale-list",
        help="the types of rescalings to do, defaults to spin_embed, for ablations do spin_embed,copy,zeros",
        type=lambda s: s.split(","),
        default="spin_embed",
    )
    parser.add_argument(
        "--subsample", help="how much to subsample the trajectories", type=int, default=1
    )
    parser.add_argument("-N", help="spatial size", type=int, default=64)
    # defaults for diff_burgers are 1.5 and -1.5, we scale them by D, so for these defaults we
    # unscale by 2 so that D=2 uses the defaults
    parser.add_argument(
        "--diffusion-coef", help="the diffusion coefficient", type=float, default=1.5 / 2
    )
    parser.add_argument(
        "--convection-coef", help="the convection coefficient", type=float, default=-1.5 / 2
    )
    parser.add_argument(
        "--n-timesteps", help="the number of timesteps in each trajectory", type=int, default=50
    )
    parser.add_argument(
        "--residual",
        help="learn the residual of the heat equation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--pretrain-batch",
        type=int,
        default=8,
        help="batch size for low dimensional pretraining, defaults for 2 GPUs",
    )
    parser.add_argument(
        "--finetune-batch",
        type=int,
        default=4,
        help="batch size for higher dimensional finetuning, defaults are for 2 GPUs",
    )
    parser.add_argument(
        "--pretrain-lr-range",
        help="benchmark trained model over the lr",
        type=lambda s: tuple(float(x) for x in s.split(",")) if isinstance(s, str) else None,
        default=None,
    )
    parser.add_argument(
        "--finetune-lr-range",
        help="benchmark tuned model over the lr",
        type=lambda s: tuple(float(x) for x in s.split(",")) if isinstance(s, str) else None,
        default=None,
    )
    # need do to --train-wandb or --tune-wandb to activate
    parser.add_argument("--wandb-project", help="the wandb project", type=str, default="burgers")

    return parser.parse_args()


# MAIN
args = handleArgs()
if args.wandb:
    print("Use --train-wandb or --test-wandb to control these individually. Exiting.")
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
free_filters_dict, upsample_filters_dict = anyd_helpers.generate_filters(full_D_range, [2])

train_kwargs = {
    "train_loss_f": geom.Losses.NRMSE,
    "batch_size": args.pretrain_batch,
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
    "batch_size": args.finetune_batch,
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
    "batch_size": args.finetune_batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": args.images_dir,  # currently ignored
    "verbose": args.verbose,
    "is_wandb": finetune_lr,
    "wandb_project": args.wandb_project,
    "wandb_entity": args.wandb_entity,
}

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, _ = get_data_d(
        D,
        args.N,
        args.diffusion_coef,
        args.convection_coef,
        args.n_timesteps,
        args.subsample,
        args.n_train,
        subkey,
        pathlib.Path(args.data) / "train",
    )
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
                        "lr": {2: {8: 1e-4}, 3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 8: 1e-4}},
                        # D=3, for all of them (and tuning) its just 1e-4
                        **train_kwargs,
                    },
                    {  # train_kwargs
                        "lr": {2: {8: 1e-4}, 3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 8: 1e-4}},
                        **baseline_kwargs,
                    },
                    {  # tune and eval kwargs
                        "lr": {2: {3: {0: 1e-4, 1: 1e-4, 4: 1e-4, 8: 1e-4}}},
                        "conv_filters_dict": free_filters_dict,
                        "upsample_filters_dict": upsample_filters_dict,
                        **test_kwargs,
                    },
                ),
            ]
        )

    model_list_d[D] = model_list

# TODO: old training time saved model doesn't have time
# will want to re-run, but for now this is 2gpus, batch=8 1587.40561104 1gpu batch=8 1026.90299463


# extended lambda function
def get_data_lambda(
    D: int, n_train: int, n_val: int, n_test: int, batch_size: int, key: PRNGKeyArray
) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    float,
]:
    return get_data(
        D,
        args.N,
        args.diffusion_coef,
        args.convection_coef,
        args.n_timesteps,
        args.subsample,
        n_train,
        n_val,
        n_test,
        batch_size,
        key,
        args.data,
    )


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
    {2: args.pretrain_batch, 3: args.finetune_batch},
)
