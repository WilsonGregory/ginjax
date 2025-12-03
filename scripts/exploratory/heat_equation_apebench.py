from __future__ import annotations
import argparse
import functools as ft
import math
import matplotlib.pyplot as plt
import pathlib
import time
from typing_extensions import Self

import jax
import jax.numpy as jnp
import jax.random as random
import equinox as eqx
import optax
import apebench
import exponax

import ginjax.geometric as geom
import ginjax.ml as ml
import ginjax.models as models
import ginjax.utils as utils
import ginjax.data as gc_data


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
    for i, idx in enumerate(idxs):
        print(f"{D},{batch}: {i}/{len(idxs)}")
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


def get_data(
    data_dir: str,
    train_D: int,
    range_test_D: list[int],
    N: int,
    n_train: int,
    n_val: int,
    n_test: int,
    subsample: int,
    past_steps: int,
    key: jax.Array,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    list[geom.MultiImage],
    list[geom.MultiImage],
]:
    if train_D != range_test_D[0]:
        raise ValueError()

    is_torus = True
    n_timesteps = 2  # 50?
    n_timesteps_int = n_timesteps * subsample  # integrator time steps
    n_warmup_steps = 1  # in 3D, seems like there is an initial problem
    dt = 1

    diffusion_coef = 2.0  # default for difficulty.Diffusion

    diffusion_stepper = exponax.stepper.Diffusion(
        train_D, domain_extent=N, num_points=N, dt=dt, diffusivity=diffusion_coef
    )

    # train_name = f"{train_D}D_{scenario}_N{N}_timesteps{n_timesteps_int}_diffusion{diffusion_coef * scaler(train_D)}"
    print(f"Generating train data D={train_D}")
    key, subkey = random.split(key)
    cpu = jax.devices("cpu")[0]

    train_ic_gen = exponax.ic.GaussianRandomField(train_D, zero_mean=True, std_one=True)
    # train_x0 = exponax.build_ic_set(train_ic_gen, num_points=N, num_samples=n_train, key=subkey)
    train_x0 = random.normal(subkey, shape=(n_train, 1) + (N,) * train_D)
    train_xt = heat_step(train_D, train_x0[:, 0], dt, diffusion_coef, is_torus)[:, None]
    # train_xt = diffusion_stepper.step(train_x0)  # (batch,channels,spatial)

    train_x0 = jax.device_put(train_x0, cpu)
    train_xt = jax.device_put(train_xt, cpu)

    train_X = geom.MultiImage({(0, 0): train_x0}, train_D, is_torus)
    train_Y = geom.MultiImage({(0, 0): train_xt}, train_D, is_torus)

    key, subkey = random.split(key)
    val_x0 = random.normal(subkey, shape=(n_val, 1) + (N,) * train_D)
    # val_x0 = exponax.build_ic_set(train_ic_gen, num_points=N, num_samples=n_val, key=subkey)
    val_xt = heat_step(train_D, val_x0[:, 0], dt, diffusion_coef, is_torus)[:, None]
    # val_xt = diffusion_stepper.step(val_x0)  # (batch,spatial)

    val_x0 = jax.device_put(val_x0, cpu)
    val_xt = jax.device_put(val_xt, cpu)

    val_X = geom.MultiImage({(0, 0): val_x0}, train_D, is_torus)
    val_Y = geom.MultiImage({(0, 0): val_xt}, train_D, is_torus)

    test_Xs = []
    test_Ys = []
    for D in range_test_D:
        # if D=3, N=128, baseline timesteps=50, subsample=8, that takes 10Gb of memory, so split it up

        test_ic_gen = exponax.ic.GaussianRandomField(D, zero_mean=True, std_one=True)
        key, subkey = random.split(key)
        # test_x0 = exponax.build_ic_set(test_ic_gen, num_points=N, num_samples=n_test, key=subkey)
        test_x0 = random.normal(subkey, shape=(n_test, 1) + (N,) * D)
        test_xt = heat_step(D, test_x0[:, 0], dt, diffusion_coef, is_torus)[:, None]
        # test_xt = diffusion_stepper.step(test_x0)  # (batch,spatial)

        test_x0 = jax.device_put(test_x0, cpu)
        test_xt = jax.device_put(test_xt, cpu)

        test_X = geom.MultiImage({(0, 0): test_x0}, D, is_torus)
        test_Y = geom.MultiImage({(0, 0): test_xt}, D, is_torus)

        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    return train_X, train_Y, val_X, val_Y, test_Xs, test_Ys


def get_data_old(
    data_dir: str,
    train_D: int,
    range_test_D: list[int],
    N: int,
    n_train: int,
    n_val: int,
    n_test: int,
    subsample: int,
    past_steps: int,
    key: jax.Array,
) -> tuple[
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    geom.MultiImage,
    list[geom.MultiImage],
    list[geom.MultiImage],
]:
    if train_D != range_test_D[0]:
        raise ValueError()

    is_torus = True
    n_timesteps = 2  # 50?
    n_timesteps_int = n_timesteps * subsample  # integrator time steps
    n_warmup_steps = 1  # in 3D, seems like there is an initial problem
    scenario = "diff_diff"

    # apebench.scenarios.physical.Diffusion()
    # apebench.scenarios.normalized.Diffusion()
    # apebench.scenarios.difficulty.Diffusion()

    diffusion_coef = 4  # default for difficulty.Diffusion
    # diffusion_coef = 0.00008  # default for norm.Diffusion is 0.0008
    # diffusion_coef = 0.00008  # default for physical.Diffusion is 0.00008
    scaler = lambda d: d

    train_name = f"{train_D}D_{scenario}_N{N}_timesteps{n_timesteps_int}_diffusion{diffusion_coef * scaler(train_D)}"
    print(train_name)
    train_path = pathlib.Path(f"{data_dir}") / f"{train_name}_train.npy"
    val_path = pathlib.Path(f"{data_dir}") / f"{train_name}_test.npy"
    if not train_path.is_file() or not val_path.is_file():
        print(f"Generating train data D={train_D}")
        key, subkey = random.split(key)
        train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

        apebench.scraper.scrape_data_and_metadata(
            data_dir,
            scenario=scenario,
            name=train_name,
            num_spatial_dims=train_D,
            num_points=N,
            num_warmup_steps=n_warmup_steps,
            num_train_samples=n_train,
            num_test_samples=n_val,
            train_seed=int(train_seed),
            test_seed=int(test_seed),
            train_temporal_horizon=n_timesteps_int - 1,
            test_temporal_horizon=n_timesteps_int - 1,
            # diffusion_coef=diffusion_coef * scaler(train_D),
            # diffusion_alpha=diffusion_coef * scaler(train_D),
            diffusion_gamma=diffusion_coef * scaler(train_D),
        )

    cpu = jax.devices("cpu")[0]
    # (batch,timesteps,tensor,spatial) -> (batch,timesteps,spatial)
    train_data = jax.device_put(jnp.load(train_path)[:, ::subsample, 0], cpu)
    val_data = jax.device_put(jnp.load(val_path)[:, ::subsample, 0], cpu)
    # subsample here for memory efficiency

    train_data /= jnp.std(train_data[:, :1])
    val_data /= jnp.std(train_data[:, :1])

    constant_fields = geom.MultiImage({}, train_D, is_torus)
    train_X, train_Y = gc_data.batch_time_series(
        geom.MultiImage({(0, 0): train_data}, train_D, is_torus),
        constant_fields,
        n_timesteps,
        past_steps,
        1,
    )
    val_X, val_Y = gc_data.batch_time_series(
        geom.MultiImage({(0, 0): val_data}, train_D, is_torus),
        constant_fields,
        n_timesteps,
        past_steps,
        1,
    )

    test_Xs = []
    test_Ys = []
    key, subkey = random.split(key)
    test_seeds = random.randint(subkey, shape=(len(range_test_D),), minval=0, maxval=10000)
    for D, test_seed in zip(range_test_D, test_seeds):
        # if D=3, N=128, baseline timesteps=50, subsample=8, that takes 10Gb of memory, so split it up
        batch = 1 if D == 3 else n_test

        test_data = jax.device_put(jnp.zeros((0, n_timesteps) + (N,) * D), cpu)
        for i in range(n_test // batch):
            # diff_burgers scales diffusion_gamma and convection_delta by D, so we unscale them so
            # that the equation is the same across dimensions.

            # a little awkward because it will say test_test at the end
            test_name = f"{D}D_{scenario}_N{N}_timesteps{n_timesteps_int}_diffusion{diffusion_coef * scaler(D)}_i{i}_test"
            test_path = pathlib.Path(f"{data_dir}") / f"{test_name}_test.npy"
            if not test_path.is_file():
                print(f"Generating test data, D={D}")
                key, subkey = random.split(key)
                train_seed, test_seed = random.randint(subkey, shape=(2,), minval=0, maxval=10000)

                apebench.scraper.scrape_data_and_metadata(
                    data_dir,
                    scenario=scenario,
                    name=test_name,
                    num_spatial_dims=D,
                    num_points=N,
                    num_warmup_steps=n_warmup_steps,
                    num_train_samples=0,
                    num_test_samples=batch,
                    test_seed=int(test_seed),
                    train_temporal_horizon=n_timesteps_int - 1,
                    test_temporal_horizon=n_timesteps_int - 1,
                    # diffusion_coef=diffusion_coef * scaler(D),  # may have to scale relative to D
                    # diffusion_alpha=diffusion_coef * scaler(D),  # may have to scale relative to D
                    diffusion_gamma=diffusion_coef * scaler(D),  # may have to scale relative to D
                )

            # subsample here for memory efficiency
            test_data_i = jax.device_put(jnp.load(test_path)[:, ::subsample, 0], cpu)
            test_data = jnp.concatenate([test_data, test_data_i], axis=0)

        test_data /= jnp.std(test_data[:, :1])

        print(f"{D}, ({jnp.mean(test_data[:, 0]):.4e}, {jnp.std(test_data[:,0]):.3f})", end=" ")
        for i in range(1, n_timesteps):
            prev = jnp.std(test_data[:, i - 1])
            cur = jnp.std(test_data[:, i])
            print(f"({jnp.std(test_data[:,i]):.4e}, {(prev - cur)*100/ prev:.1f}%)", end=" ")

            # print(f"({jnp.mean(test_data[:, i]):.4e}, {jnp.std(test_data[:,i]):.3f})", end=" ")

            # print(
            #     test_data.shape,
            #     jnp.mean(jnp.abs(test_data[:, 0] - test_data[:, 1])),
            #     jnp.mean(jnp.abs(test_data[:, 0] - test_data[:, 1]) / jnp.abs(test_data[:, 1])),
            #     jnp.mean(jnp.abs(test_data[:, 0])),
            #     jnp.mean(jnp.abs(test_data[:, 1])),
            # )

            # print(
            #     test_data.shape,
            #     jnp.mean(jnp.abs(test_data[:, 1] - test_data[:, 2])),
            #     jnp.mean(jnp.abs(test_data[:, 1] - test_data[:, 2]) / jnp.abs(test_data[:, 2])),
            #     jnp.mean(jnp.abs(test_data[:, 1])),
            #     jnp.mean(jnp.abs(test_data[:, 2])),
            # )

        print("")

        constant_fields = geom.MultiImage({}, D, is_torus)

        test_X, test_Y = gc_data.batch_time_series(
            geom.MultiImage({(0, 0): test_data}, D, is_torus),
            constant_fields,
            n_timesteps,
            past_steps,
            1,
        )

        test_Xs.append(test_X)
        test_Ys.append(test_Y)

    return train_X, train_Y, val_X, val_Y, test_Xs, test_Ys


def plot_multi_image(
    input_multi_image: geom.MultiImage,
    actual_multi_image: geom.MultiImage,
    test_multi_image: geom.MultiImage,
    save_loc: str,
    title: str = "",
):
    """
    Plot vector x and y components of two MultiImages, and the differences between them. Each row
    is a component, and the columns are actual, test, and diff. If the image is 3D, plot the middle
    slice.

    args:
        input_multi_image_full: the input image, with past_steps number of timesteps
        actual_multi_image: the ground truth MultiImage
        test_multi_image: the predicted MultiImage
        save_loc: file location to save the image
        title: additional str to add to title, will be "test {title} {col}"
            "actual {title} {col}"
    """
    print(
        f"Printed image relative error: {ml.l2_rel_error(test_multi_image, actual_multi_image):.4f}%"
    )

    if input_multi_image.get_n_leading() == 2:
        input_multi_image = input_multi_image.get_one(keepdims=False)

    if actual_multi_image.get_n_leading() == 2:
        actual_multi_image = actual_multi_image.get_one(keepdims=False)

    if test_multi_image.get_n_leading() == 2:
        test_multi_image = test_multi_image.get_one(keepdims=False)

    # images now no longer have batch dimension

    N = input_multi_image.get_spatial_dims()[0]
    if input_multi_image.D == 3:
        img_arr = jnp.concatenate(
            [
                input_multi_image[((False,), 0)][:, :, N // 2],
                test_multi_image[((False,), 0)][:, :, N // 2],
                actual_multi_image[((False,), 0)][:, :, N // 2],
            ]
        )
    else:
        img_arr = jnp.concatenate(
            [
                input_multi_image.to_vector(),
                test_multi_image.to_vector(),
                actual_multi_image.to_vector(),
            ]
        )

    vmax = float(jnp.max(jnp.abs(img_arr)))
    vmin = -1 * vmax

    nrows = test_multi_image.D
    ncols = 4
    # figsize is 6 per col, 6 per row, (cols,rows)
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(6 * ncols, 6 * nrows))
    for component in range(test_multi_image.D):
        comp_name = ["x", "y", "z"][component]

        input_image = input_multi_image.get_component(component).to_images()[0]
        if input_image.D == 3:
            input_image = geom.GeometricImage(input_image.data[N // 2], input_image.parity, 2)

        input_image.plot(
            axes[component, 0],
            title=f"input {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )

        actual_image = actual_multi_image.get_component(component).to_images()[0]
        test_image = test_multi_image.get_component(component).to_images()[0]

        if actual_image.D == 3:
            actual_image = geom.GeometricImage(actual_image.data[N // 2], actual_image.parity, 2)
        if test_image.D == 3:
            test_image = geom.GeometricImage(test_image.data[N // 2], test_image.parity, 2)

        diff = actual_image - test_image
        diff_max = float(jnp.max(jnp.abs(diff.data)))
        diff_l2 = jnp.mean(diff.data**2)
        diff_rel_err = jnp.mean(jnp.abs(diff.data / actual_image.data)) * 100
        diff_title = f"diff (l2: {diff_l2:.3e}, rel. err: {diff_rel_err:.2f}%)"

        actual_image.plot(
            axes[component, 1],
            title=f"output {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        test_image.plot(
            axes[component, 2],
            title=f"pred {title} {comp_name}",
            vmin=vmin,
            vmax=vmax,
            colorbar=True,
        )
        diff.plot(
            axes[component, 3],
            title=diff_title,
            vmin=-diff_max,
            vmax=diff_max,
            colorbar=True,
        )

    plt.tight_layout()
    plt.savefig(save_loc)
    plt.close(fig)


class TwoLayerModel(models.AnyDimensionalModel):
    layers: list[models.ConvBlock]

    D: int = eqx.field(static=True)
    input_keys: geom.Signature = eqx.field(static=True)
    target_keys: geom.Signature = eqx.field(static=True)
    depth: int = eqx.field(static=True)
    use_bias: bool | str = eqx.field(static=True)

    def __init__(
        self: Self,
        input_keys: geom.Signature,
        target_keys: geom.Signature,
        conv_filters: geom.MultiImage,
        depth: int,
        use_bias: bool | str,
        key: jax.Array,
    ) -> None:
        self.D = conv_filters.D
        self.input_keys = input_keys
        self.target_keys = target_keys
        self.depth = depth
        self.use_bias = use_bias

        mid_keys = geom.signature_union(input_keys, target_keys, depth)

        key, subkey1, subkey2 = random.split(key, num=3)
        self.layers = [
            models.ConvBlock(
                D, input_keys, mid_keys, use_bias, "gelu", True, conv_filters, key=subkey1
            ),
            models.ConvBlock(
                D, mid_keys, target_keys, use_bias, None, True, conv_filters, key=subkey2
            ),
        ]

    def convertD(
        self: Self, conv_filters: geom.MultiImage, rescale: bool, key: jax.Array, **kwargs
    ) -> Self:
        """
        Construct a new model with filters in a higher dimension.

        args:
            conv_filters: the new conv filters we are swapping to, probably in a higher dimension
            rescale: whether to force the sum of the filters in the new dimension to be equal
            key: key to initialize the weights, since they are overruled it won't matter

        returns:
            a new model with new filters but the old weights
        """
        new_model = self.__class__(
            self.input_keys,
            self.target_keys,
            conv_filters,
            self.depth,
            self.use_bias,
            key,
        )

        return self.transfer_weights(new_model, rescale)

    def __call__(
        self: Self, x: geom.MultiImage, aux_data: eqx.nn.State | None = None
    ) -> tuple[geom.MultiImage, eqx.nn.State | None]:
        in_x = x
        for layer in self.layers:
            x, aux_data = layer(x, aux_data)

        return x, aux_data


@eqx.filter_jit
def map_residual(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[geom.MultiImage, eqx.nn.State | None]:
    residual, aux_data = jax.vmap(model, in_axes=(0, None), out_axes=(0, None), axis_name="batch")(
        multi_image_x, aux_data
    )

    # add the last timestep to the residual
    pred_y = residual.empty()
    for ((k, parity), img_in), img_resid in zip(multi_image_x.items(), residual.values()):
        pred_y.append(k, parity, img_in[:, -1:] + img_resid)

    return pred_y, aux_data


@eqx.filter_jit
def map_and_loss_rel_error(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    """
    Calculates both the smse_loss and the rel_error.
    """
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    loss = ml.smse_loss(pred_y, multi_image_y)
    rel_error = ml.l2_rel_error(pred_y, multi_image_y, eps=1e-8)
    return jnp.stack([loss, rel_error]), aux_data


@eqx.filter_jit
def map_and_loss(
    model: models.MultiImageModule,
    multi_image_x: geom.MultiImage,
    multi_image_y: geom.MultiImage,
    aux_data: eqx.nn.State | None = None,
) -> tuple[jax.Array, eqx.nn.State | None]:
    pred_y, aux_data = map_residual(model, multi_image_x, aux_data)

    return ml.smse_loss(pred_y, multi_image_y), aux_data


def train_and_eval(
    data: tuple[
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        geom.MultiImage,
        list[geom.MultiImage],
        list[geom.MultiImage],
    ],
    key: jax.Array,
    model_name: str,
    model: models.AnyDimensionalModel,
    lr: float,
    test_conv_filters: list[tuple[geom.MultiImage, geom.MultiImage]],
    batch_size: int,
    test_batch_size: int,
    epochs: int,
    save_model: str | None,
    load_model: str | None,
    images_dir: str | None,
    has_aux: bool = False,
    verbose: int = 1,
    is_wandb: bool = False,
) -> tuple[jax.Array | None, ...]:
    train_X, train_Y, val_X, val_Y, test_d_X, test_d_Y = data
    N = train_X.get_spatial_dims()[0]
    batch_stats = eqx.nn.State(model) if has_aux else None
    model_name_extended = f"{model_name}_L{train_X.get_L()}_N{N}_e{epochs}"

    print(f"Model params: {models.count_params(model):,}")

    if load_model is None:
        steps_per_epoch = int(math.ceil(train_X.get_L() / batch_size))
        key, subkey = random.split(key)
        trained_model, batch_stats, train_loss, val_loss = ml.train(
            train_X,
            train_Y,
            map_and_loss,
            model,
            subkey,
            stop_condition=ml.EpochStop(epochs, verbose=verbose),
            batch_size=batch_size,
            optimizer=optax.adamw(
                optax.warmup_cosine_decay_schedule(
                    1e-8, lr, 5 * steps_per_epoch, epochs * steps_per_epoch, 1e-7
                ),
                weight_decay=1e-5,
            ),
            validation_X=val_X,
            validation_Y=val_Y,
            aux_data=batch_stats,
            is_wandb=is_wandb,
        )

        if save_model is not None:
            # TODO: need to save batch_stats as well
            ml.save(f"{save_model}{model_name_extended}_model.eqx", trained_model)
    else:
        trained_model = ml.load(f"{load_model}{model_name_extended}_model.eqx", model)

        key, subkey1, subkey2 = random.split(key, num=3)
        train_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            train_X,
            train_Y,
            batch_size,
            subkey1,
            aux_data=batch_stats,
        )
        val_loss = ml.map_loss_in_batches(
            map_and_loss,
            trained_model,
            val_X,
            val_Y,
            batch_size,
            subkey2,
            aux_data=batch_stats,
        )

    assert isinstance(trained_model, models.AnyDimensionalModel)
    test_losses = []
    for test_X, test_Y, (conv_filters, upsample_filters) in zip(
        test_d_X, test_d_Y, test_conv_filters
    ):
        if test_X.D == train_X.D:
            trained_model_d = trained_model
        else:
            key, subkey = random.split(key)
            # rescale can be true or false when using zero_sum, the effect is small
            trained_model_d = trained_model.convertD(
                conv_filters, False, subkey, upsample_filters=upsample_filters
            )

        key, subkey = random.split(key)
        test_loss = ml.map_loss_in_batches(
            map_and_loss_rel_error,
            trained_model_d,
            test_X,
            test_Y,
            test_batch_size,
            subkey,
            aux_data=batch_stats,
        )
        print(f"Test Loss D={test_X.D}: {test_loss[0]}")
        print(f"Test Relative Error D={test_X.D}: {test_loss[1]:.4f}%")

        test_losses.append(test_loss[0])
        test_losses.append(test_loss[1])

        if images_dir:
            pred_y, _ = jax.vmap(trained_model_d, in_axes=(0, None), out_axes=(0, None))(
                test_X.get_one(), batch_stats
            )

            plot_multi_image(
                test_X.get_one(),
                test_Y.get_one(),
                pred_y.get_one(),
                f"{images_dir}{model_name_extended}_D{test_X.D}.png",
                "heat",
            )

    return train_loss, val_loss, *test_losses


def handleArgs() -> argparse.Namespace:
    parser = utils.get_common_parser()
    parser.add_argument(
        "--train-D", help="dimension of data to train on", choices=[1, 2, 3], default=1, type=int
    )
    parser.add_argument(
        "--max-test-D",
        help="maximum dimension of data to test on",
        choices=[1, 2, 3],
        default=3,
        type=int,
    )
    parser.add_argument("-N", help="spatial size", type=int, default=128)
    parser.add_argument(
        "--past-steps", help="number of past steps to use as input", type=int, default=1
    )
    parser.add_argument(
        "--subsample", help="how much to subsample the trajectories", type=int, default=8
    )
    parser.add_argument("--test-batch", help="batch size for test data", type=int, default=1)
    # need do to --wandb to activate, also need --wandb-entity your_wandb_name_here
    parser.add_argument(
        "--wandb-project", help="the wandb project", type=str, default="heat-equation-apebench"
    )

    return parser.parse_args()


# Main
args = handleArgs()
assert args.train_D <= args.max_test_D

range_test_D = list(range(args.train_D, args.max_test_D + 1))

key = random.PRNGKey(time.time_ns()) if (args.seed is None) else random.PRNGKey(args.seed)

key, subkey = random.split(key)
data = get_data(
    args.data,
    args.train_D,
    range_test_D,
    args.N,
    args.n_train,
    args.n_val,
    args.n_test,
    args.subsample,
    args.past_steps,
    subkey,
)

input_keys = data[0].get_signature()
output_keys = data[1].get_signature()

group_actions = geom.make_all_operators(args.train_D)
conv_filters = geom.get_invariant_filters(
    Ms=[3],
    ks=[0],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=geom.FilterScaling.ZERO_SUM,
)
upsample_filters = geom.get_invariant_filters(
    Ms=[2],
    ks=[0],
    parities=[0],
    D=args.train_D,
    operators=group_actions,
    scale=geom.FilterScaling.ZERO_SUM,
)

test_conv_filters = []
for D in range(args.train_D, 4):
    group_actions_d = geom.make_all_operators(D)
    conv_filters_d = geom.get_invariant_filters(
        Ms=[3],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=geom.FilterScaling.ZERO_SUM,
    )
    upsample_filters_d = geom.get_invariant_filters(
        Ms=[2],
        ks=[0],
        parities=[0],
        D=D,
        operators=group_actions_d,
        scale=geom.FilterScaling.ZERO_SUM,
    )
    test_conv_filters.append((conv_filters_d, upsample_filters_d))


train_kwargs = {
    "test_conv_filters": test_conv_filters,
    "batch_size": args.batch,
    "test_batch_size": args.batch,
    "epochs": args.epochs,
    "save_model": args.save_model,
    "load_model": args.load_model,
    "images_dir": args.images_dir,
    "verbose": args.verbose,
    "is_wandb": args.wandb,
}

key, *subkeys = random.split(key, num=10)
model_list = [
    (
        "two_layer",
        train_and_eval,
        {
            "model": TwoLayerModel(
                input_keys, output_keys, conv_filters, 10, "auto", key=subkeys[0]
            ),
            "lr": 1e-2,
            **train_kwargs,
        },
    ),
    (
        "lastStepIdentity",
        train_and_eval,
        {"model": models.LastStepIdentity(residual=True), "lr": 1, **train_kwargs},
    ),
    # (
    #     "resnet_equiv_42",
    #     train_and_eval,
    #     {
    #         "model": models.ResNet(
    #             args.train_D,
    #             input_keys,
    #             output_keys,
    #             depth=42,
    #             conv_filters=conv_filters,
    #             use_group_norm=False,
    #             key=subkeys[1],
    #         ),
    #         "lr": 7e-4,
    #         **train_kwargs,
    #     },
    # ),
]

key, subkey = random.split(key)
# Use this for benchmarking the models with known learning rates.
results = ml.benchmark(
    lambda _: data,
    model_list,
    subkey,
    "",
    [0],
    benchmark_type=ml.BENCHMARK_NONE,
    num_trials=args.n_trials,
    num_results=5,
    is_wandb=args.wandb,
    wandb_project=args.wandb_project,
    wandb_entity=args.wandb_entity,
)
