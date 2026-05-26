from __future__ import annotations
import argparse
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import jax.random as random
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils
from . import anyd_helpers


def heat_step(D: int, x0: jax.Array, t: float, k: float, is_torus: bool) -> jax.Array:
    """
    Given the initial temperature, time, and diffusion coefficient, calculate the new heat field.

    args:
        D: the dimension
        x0: the starting temperature array, shape (batch, spatial)
        t: the length of the timestep
        k: diffusion coefficient
        is_torus: whether the data lives on the torus

    returns:
        the heat field after time t
    """
    assert t >= 0
    if t == 0:
        return x0

    batch = len(x0)
    spatial_dims = x0.shape[1:]
    x0_flat = x0.reshape((batch, -1))
    meshgrid_dims = (jnp.arange(N) for N in spatial_dims)

    # (spatial_size,D)
    idxs = jnp.stack(jnp.meshgrid(*meshgrid_dims, indexing="ij"), axis=-1).reshape((-1, D))
    x1 = []
    # Effectively batching this entire dataset runs out of memory
    # It is possible there are more efficient ways of doing this.
    for i, idx in enumerate(idxs):
        # (spatial_size,D)
        if is_torus:
            idxs_diff = jnp.abs(idxs - idx[None])
            idxs_diff_wrapped = jnp.stack(spatial_dims).reshape((1, D)) - idxs_diff
            dist = jnp.where(idxs_diff < idxs_diff_wrapped, idxs_diff, idxs_diff_wrapped)
        else:
            dist = idxs - idx[None]

        # (1,spatial_size)
        heat_kernel = jnp.exp((jnp.linalg.norm(dist, axis=1) ** 2) / (-4 * k * t))[None]

        x1.append(jnp.sum(heat_kernel * x0_flat, axis=1) / ((4 * jnp.pi * k * t) ** (D / 2)))

    return jnp.stack(x1, axis=1).reshape(x0.shape)


def get_data_d(
    D: int,
    N: int,
    is_torus: bool,
    k: float,
    t: float,
    max_temp: float,
    batch: int,
    key: jax.Array,
    data_dir: pathlib.Path,
) -> tuple[geom.MultiImage, geom.MultiImage, float]:
    """
    Get an input, output data pair of heat diffusion after t timestep, diffusion coefficient k.
    The initial data is uniform on the range 0 to max_temp.

    args:
        D: the dimension of the space
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        k: diffusion coefficient
        t: size of timestep
        max_temp: max temperature
        batch: number of data points
        key: key for randomness
        data_dir: directory
        data_name: name where to save the data

    returns:
        input multi image, output multi image
    """
    data_dir = (
        data_dir / f"D{D}_N{N}_istorus{int(is_torus)}_n{batch}_k{k}_t{t}_maxtemp{max_temp}.npy"
    )

    if batch == 0:
        x0 = jnp.zeros((batch,) + (N,) * D)
        xt = jnp.zeros((batch,) + (N,) * D)
        generation_time = 0
    elif data_dir.is_file():
        dataset = jnp.load(data_dir, allow_pickle=True).item()
        x0 = dataset["x0"]
        xt = dataset["xt"]
        generation_time = dataset["generation_time"] if "generation_time" in dataset else -1
    else:
        print(f"Creating data {data_dir}...")
        start_time = time.time()
        key, subkey = random.split(key)
        x0 = random.uniform(subkey, shape=(batch,) + (N,) * D, minval=-max_temp, maxval=max_temp)
        xt = heat_step(D, x0, t, k, is_torus)
        generation_time = time.time() - start_time

        print(f"Finished in {generation_time} seconds.")
        jnp.save(data_dir, {"x0": x0, "xt": xt, "generation_time": generation_time})

    x0_img = geom.MultiImage({(0, 0): x0[:, None]}, D, is_torus)
    xt_img = geom.MultiImage({(0, 0): xt[:, None]}, D, is_torus)

    return x0_img, xt_img, generation_time


def get_data(
    D: int,
    N: int,
    is_torus: bool,
    diffusion_coef: float,
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
    Get an input, output data pair of heat diffusion after 1 timestep, specified diffusion constant.
    The initial data is uniform on the range -sqrt(3) to sqrt(3), so it has unit variance.

    args:
        D: data dimension
        N: sidelength of a cube of data
        is_torus: whether the data is on the torus
        diffusion_coeff: coefficient in the diffusion equation, equivalently the timestep
        n_train: number of training data points
        n_val: number of validation data points
        n_test: number of test data points for each test dimension
        batch_size: the training batch size
        key: key for randomness
        data_dir: location to save or load the data from

    returns:
        training, validation, and test images for input and output
    """
    max_temp = math.sqrt(3)
    t = 1
    data_dir_path = pathlib.Path(data_dir)

    key, subkey1, subkey2, subkey3 = random.split(key, num=4)
    train_x0, train_xt, train_generation_time = get_data_d(
        D, N, is_torus, diffusion_coef, t, max_temp, n_train, subkey1, data_dir_path / "train"
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
        D, N, is_torus, diffusion_coef, t, max_temp, n_val, subkey2, data_dir_path / "val"
    )
    val_dataset = ml.MultiImageDataset(val_x0, val_xt)
    val_dataloader = DataLoader(
        val_dataset,
        sampler=BatchSampler(SequentialSampler(val_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    test_x0, test_xt, _ = get_data_d(
        D, N, is_torus, diffusion_coef, t, max_temp, n_test, subkey3, data_dir_path / "test"
    )
    test_dataset = ml.MultiImageDataset(test_x0, test_xt)
    test_dataloader = DataLoader(
        test_dataset,
        sampler=BatchSampler(SequentialSampler(test_dataset), batch_size, drop_last=True),
        collate_fn=lambda x: x[0],
    )

    if n_test > 0:
        x0_std = jnp.std(test_x0[((), 0)])
        xt_std = jnp.std(test_xt[((), 0)])
        xt_resid_std = jnp.std(test_xt[((), 0)] - test_x0[((), 0)])
        print(f"D={D}, x0:{x0_std:.3e}, xt:{xt_std:.3e}, xt_resid:{xt_resid_std:.3e}")

    return train_dataloader, val_dataloader, test_dataloader, train_generation_time


# an example of currently used script
# CUDA_VISIBLE_DEVICES=6 time python3 -m scripts.heat_equation --data /data/wgregor4/heat_equation/
# --n-test 128 --n-val 128 --n-train 128 --train-D-range 1 --test-D-range 2
# --model-dir /data/wgregor4/runs/heat_equation/
def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--n-tune-range",
        help="the number of data points in the tuning set",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="0,1,4,32,128",
    )
    parser.add_argument(
        "--train-D-range",
        help="a comma separated list of range of dims to train over, e.g. 1,2",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="1,2",
    )
    parser.add_argument(
        "--test-D-range",
        help="a comma separated list of range of dims to test over, e.g. 1,2",
        type=lambda s: tuple(int(x) for x in s.split(",")),
        default="2,3",
    )
    parser.add_argument("-N", help="spatial size", type=int, default=64)
    parser.add_argument(
        "--diffusion-coef", help="the diffusion coefficient", type=float, default=1.0
    )
    parser.add_argument(
        "--residual",
        help="learn the residual of the heat equation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--find-train-lr",
        help="benchmark trained model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--find-tune-lr",
        help="benchmark tuned model over the lr",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    # need do to --train-wandb or --tune-wandb to activate
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation"
    )
    parser.add_argument(
        "--train-wandb",
        help="whether to use wandb during training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--tune-wandb",
        help="whether to use wandb during tuning",
        action=argparse.BooleanOptionalAction,
        default=False,
    )

    return parser.parse_args()


# MAIN
args = handleArgs()
if args.wandb:
    print("Use --train-wandb or --test-wandb to control these individually. Exiting.")
    exit()

if args.load_model or args.save_model:
    print("Use --model-dir and possibly --overwrite-save-model instead of --save-model")
    exit()

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)
lr_range = [1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2]

max_pixel_l1 = 2
M = 5
n_results = 5

train_kwargs = {
    "train_loss_f": geom.Losses.SMSE,
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": None,
    "verbose": args.verbose,
    "is_wandb": args.train_wandb,
}

test_kwargs = {
    "train_loss_f": geom.Losses.SMSE,
    "residual": args.residual,
    "batch_size": args.batch,
    "epochs": args.epochs,
    "model_dir": pathlib.Path(args.model_dir) if args.model_dir else None,
    "overwrite_save_model": args.overwrite_save_model,
    "images_dir": args.images_dir,  # currently ignored
    "verbose": args.verbose,
    "is_wandb": args.tune_wandb,
}

one_filters_dict = {}
normalize_filters_dict = {}
gaussian_filters_dict = {}
upsample_filters_dict = {}
free_filters_dict = {}

full_D_range = tuple(set(args.train_D_range).union(set(args.test_D_range)))
for D in full_D_range:
    group_actions = geom.make_all_operators(D)
    one_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    normalize_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.NORMALIZE,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    gaussian_filters_dict[D] = geom.get_invariant_filters(
        Ms=[M],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.GAUSSIAN,
        max_pixel_l1=max_pixel_l1,
        combine_equal_l1=True,
    )
    upsample_filters_dict[D] = geom.get_invariant_filters(
        Ms=[2],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,  # for N=2, all pixels are equidistant
    )
    free_filters_dict[D] = geom.get_invariant_filters(
        Ms=[3],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions,
        scale=geom.FilterScaling.ONE,
    )

print("Define the models!")
model_list_d = {}
for D in full_D_range:
    key, subkey = random.split(key)
    train_x0, train_xt, _ = get_data_d(
        D,
        args.N,
        True,
        args.diffusion_coef,
        1,
        math.sqrt(3),
        args.n_train,
        subkey,
        pathlib.Path(args.data) / "train",
    )
    input_keys = train_x0.get_signature()
    output_keys = train_xt.get_signature()

    key, *subkeys = random.split(key, num=10)
    model_list = [
        # (
        #     f"two_layer_gaussian_scaling_D{D}",
        #     {
        #         "model": models.SimpleConvSeries(
        #             input_keys,
        #             output_keys,
        #             gaussian_filters_dict[D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             key=subkeys[0],
        #         ),
        #         "lr": 5e-2,  # best for all dimensions, all n_tune
        #         **train_kwargs,
        #     },
        #     {
        #         "lr": {(1, 2): 1e-2, (1, 3): 5e-2, (2, 3): 1e-2},
        #         # n_tune has only a small effect on the error difference, simplicity use n=1 value
        #         # it is typically lower so this strategy is conservative
        #         # (1,2): (n=1,1e-2) (n=4,1e-2) (n=32,1e-2) (n=128,1e-2)
        #         # (1,3): (n=1,5e-2) (n=4,5e-2) (n=32,5e-2) (n=128,5e-2) (pretty large gap for n=1,4)
        #         # (2,3): (n=1,1e-2) (n=4,1e-2) (n=32,5e-2) (n=128,5e-2) can do 1e-2
        #         "rescale": geom.Rescaling.VOLUME,
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"two_layer_one_scaling_D{D}",
        #     {
        #         "model": models.SimpleConvSeries(
        #             input_keys,
        #             output_keys,
        #             one_filters_dict[D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             activation_f="gelu",
        #             key=subkeys[0],
        #         ),
        #         "lr": 5e-2,  # best for all dimensions, all n_tune
        #         **train_kwargs,
        #     },
        #     {
        #         "lr": {(1, 2): 5e-2, (1, 3): 5e-2, (2, 3): 1e-2},
        #         # n_tune has only a small effect on the error difference, simplicity use n=1 value
        #         # it is typically lower so this strategy is conservative
        #         # (1,2): (n=1,1e-2) (n=4,1e-2) (n=32,1e-2) (n=128,1e-2)
        #         # (1,3): (n=1,5e-2) (n=4,5e-2) (n=32,5e-2) (n=128,5e-2) (pretty large gap for n=1,4)
        #         # (2,3): (n=1,1e-2) (n=4,1e-2) (n=32,5e-2) (n=128,5e-2) can do 1e-2
        #         "rescale": geom.Rescaling.COMPATIBILITY,
        #         "conv_filters_dict": one_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"two_layer_free_filters_D{D}",
        #     {
        #         "model": models.SimpleConvSeries(
        #             input_keys,
        #             output_keys,
        #             free_filters_dict[D],
        #             width=10,
        #             depth=2,
        #             use_bias=False,
        #             activation_f="gelu",
        #             key=subkeys[0],
        #         ),
        #         "lr": 5e-2,  # best for all dimensions, all n_tune
        #         **train_kwargs,
        #     },
        #     {
        #         "lr": {(1, 2): 5e-2, (1, 3): 5e-2, (2, 3): 1e-2},
        #         # n_tune has only a small effect on the error difference, simplicity use n=1 value
        #         # it is typically lower so this strategy is conservative
        #         # (1,2): (n=1,1e-2) (n=4,1e-2) (n=32,1e-2) (n=128,1e-2)
        #         # (1,3): (n=1,5e-2) (n=4,5e-2) (n=32,5e-2) (n=128,5e-2) (pretty large gap for n=1,4)
        #         # (2,3): (n=1,1e-2) (n=4,1e-2) (n=32,5e-2) (n=128,5e-2) can do 1e-2
        #         "rescale": geom.Rescaling.COMPAT_FLEX,
        #         "conv_filters_dict": free_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     "lastStepIdentity",
        #     train_and_eval,
        #     {
        #         "model": models.LastStepIdentity(residual=args.residual),
        #         "lr": 1,
        #         "conv_filters_dict": normalize_filters_dict,
        #         **train_kwargs,
        #     },
        # ),
        # (
        #     f"resnet_equiv_42_gaussian_scaling_D{D}",
        #     {  # train kwargs
        #         "model": models.ResNet(
        #             D,
        #             input_keys,
        #             output_keys,
        #             depth=42,
        #             conv_filters=gaussian_filters_dict[D],
        #             use_group_norm=True,
        #             key=subkeys[1],
        #         ),
        #         "lr": {1: 5e-3, 2: 5e-4, 3: 1e-3}[D],
        #         # (D=1,n=128,5e-3) (D=2,n=1,5e-4) (D=2,n=4,5e-4) (D=2,n=32,5e-4) (D=2,n=128,5e-4)
        #         # (D=3,n=1,1e-3) (D=3,n=4,1e-3) (D=3,n=32,5e-4) (D=3,n=128,1e-3)
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": {(1, 2): 1e-4, (1, 3): 5e-5, (2, 3): 5e-5},
        #         # (1,2): (n=1,1e-4) (n=4,1e-4) (n=32,1e-3) (n=128,1e-3)
        #         # (1,3): (n=1,5e-5) (n=4,5e-5) (n=32,5e-4) (n=128,5e-3)
        #         # (2,3): (n=1,5e-5) (n=4,5e-5) (n=32,1e-4) (n=128,1e-4)
        #         "conv_filters_dict": gaussian_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"unetBase_equiv48_gaussian_scaling_D{D}",
        #     {  # train_kwargs
        #         "model": models.UNet(
        #             D,
        #             input_keys,
        #             output_keys,
        #             depth=48,
        #             activation_f=jax.nn.gelu,
        #             conv_filters=gaussian_filters_dict[D],
        #             upsample_filters=upsample_filters_dict[D],
        #             key=subkeys[2],
        #         ),
        #         "lr": 1e-3,
        #         # D=1 (n=128,1e-3)
        #         # D=2 (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,5e-4 although 1e-3 is close?)
        #         # D=3 (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,1e-3)
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": {(1, 2): 1e-3, (1, 3): 1e-3, (2, 3): 1e-3},
        #         # (1,2) (n=1,1e-3) (n=4,1e-3) (n=32,5e-3) (n=128,5e-4 or 1e-3)
        #         # (1,3) (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,5e-3,1e-3,5e-4)
        #         # (2,3) (n=1,1e-3) (n=4,1e-3) (n=32,1e-3) (n=128,1e-3 works)
        #         "conv_filters_dict": gaussian_filters_dict,
        #         "upsample_filters_dict": upsample_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        # (
        #     f"unetBase_equiv48_one_scaling_D{D}",
        #     {  # train_kwargs
        #         "model": models.UNet(
        #             D,
        #             input_keys,
        #             output_keys,
        #             depth=48,
        #             activation_f=jax.nn.gelu,
        #             conv_filters=one_filters_dict[D],
        #             upsample_filters=upsample_filters_dict[D],
        #             key=subkeys[2],
        #         ),
        #         "lr": {1: 1e-3, 2: 1e-4, 3: 1e-4}[D],
        #         # D=1 (n=128,1e-3)
        #         # D=2 (n=128,1e-4)
        #         **train_kwargs,
        #     },
        #     {  # tune and eval kwargs
        #         "lr": {(1, 2): 1e-3, (1, 3): 1e-3, (2, 3): 1e-3},
        #         # (1,2)
        #         # (1,3)
        #         # (2,3)
        #         "rescale": geom.Rescaling.COMPATIBILITY,
        #         "conv_filters_dict": one_filters_dict,
        #         "upsample_filters_dict": upsample_filters_dict,
        #         **test_kwargs,
        #     },
        # ),
        (
            f"unetBase_equiv48_free_filters_D{D}",
            {  # train_kwargs
                "model": models.UNet(
                    D,
                    input_keys,
                    output_keys,
                    depth=48,
                    activation_f=jax.nn.gelu,
                    conv_filters=free_filters_dict[D],
                    upsample_filters=upsample_filters_dict[D],
                    key=subkeys[2],
                ),
                "lr": {
                    False: {1: {128: 1e-3}, 2: {0: 1e-5, 1: 1e-5, 4: 1e-5, 32: 5e-4, 128: 5e-4}},
                    True: {
                        1: {128: 5e-4},
                        2: {0: 1e-3, 1: 1e-3, 4: 1e-3, 32: 5e-4, 128: 1e-4},
                    },
                }[args.residual],
                **train_kwargs,
            },
            {  # tune and eval kwargs
                "lr": {
                    False: {(1, 2): {0: 5e-4, 1: 5e-4, 4: 5e-4, 32: 1e-3, 128: 1e-3}},
                    True: {(1, 2): {0: 1e-4, 1: 1e-4, 4: 1e-4, 32: 1e-4, 128: 5e-4}},
                }[args.residual],
                "rescale": geom.Rescaling.COMPAT_FLEX,
                "conv_filters_dict": free_filters_dict,
                "upsample_filters_dict": upsample_filters_dict,
                **test_kwargs,
            },
        ),
    ]
    model_list_d[D] = model_list


# extended lambda function
def get_data_lambda(D: int, n_train: int, n_val: int, n_test: int, key: jax.Array) -> tuple[
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    DataLoader[ml.MultiImageDataset],
    float,
]:
    return get_data(
        D, args.N, True, args.diffusion_coef, n_train, n_val, n_test, args.batch, key, args.data
    )


key, subkey = random.split(key)
anyd_helpers.run_anyd(
    args.train_D_range,
    args.test_D_range,
    args.n_tune_range,
    args,
    subkey,
    get_data_lambda,
    model_list_d,
    lr_range,
)
