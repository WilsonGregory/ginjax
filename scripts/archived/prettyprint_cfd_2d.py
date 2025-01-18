import sys
import argparse
import matplotlib.pyplot as plt

import jax.numpy as jnp

import geometricconvolutions.geometric as geom
import geometricconvolutions.utils as utils


def handleArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-n_train", help="number of training trajectories", type=int, default=100)
    parser.add_argument(
        "-n_val",
        help="number of validation trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument(
        "-n_test",
        help="number of testing trajectories, defaults to batch",
        type=int,
        default=None,
    )
    parser.add_argument("-t", "--n_trials", help="number of trials to run", type=int, default=1)
    parser.add_argument("-seed", help="the random number seed", type=int, default=None)
    parser.add_argument("--load_M01", help="file name to load params from", type=str, default=None)
    parser.add_argument("--load_M10", help="file name to load params from", type=str, default=None)
    parser.add_argument(
        "-images_dir",
        help="directory to save images, or None to not save",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--plot_component",
        help="which component to plot, one of 0-3",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
    )
    parser.add_argument(
        "--rollout_steps",
        help="number of steps to rollout in test",
        type=int,
        default=5,
    )

    return parser.parse_args()


def process_results(results):
    rollout_res = results[..., 3:]
    non_rollout_res = jnp.concatenate(
        [results[..., :3], jnp.sum(rollout_res, axis=-1, keepdims=True)], axis=-1
    )
    mean_results = jnp.mean(
        non_rollout_res, axis=0
    )  # mean over trials (benchmark_vals,models,outputs)
    std_results = jnp.std(non_rollout_res, axis=0)  # std over trials

    return mean_results, std_results, rollout_res


# Main
args = handleArgs(sys.argv)

results_M01 = jnp.load(args.load_M01 + "results.npy")
results_M10 = jnp.load(args.load_M10 + "results.npy")

mean_M01, std_M01, rollout_M01 = process_results(results_M01)
mean_M10, std_M10, rollout_M10 = process_results(results_M10)

plot_mapping = {
    "dil_resnet64": ("DilResNet64", "blue", "o", "dashed"),
    "dil_resnet_equiv20": ("DilResNet20 (E)", "blue", "o", "dotted"),
    "dil_resnet_equiv48": ("DilResNet48 (E)", "blue", "o", "solid"),
    "resnet": ("ResNet128", "red", "s", "dashed"),
    "resnet_equiv_groupnorm_42": ("ResNet42 (E)", "red", "s", "dotted"),
    "resnet_equiv_groupnorm_100": ("ResNet100 (E)", "red", "s", "solid"),
    "unetBase": ("UNet64 Norm", "green", "P", "dashed"),
    "unetBase_equiv20": ("UNet20 Norm (E)", "green", "P", "dotted"),
    "unetBase_equiv48": ("UNet48 Norm (E)", "green", "P", "solid"),
    "unet2015": ("UNet64", "orange", "*", "dashed"),
    "unet2015_equiv20": ("Unet20 (E)", "orange", "*", "dotted"),
    "unet2015_equiv48": ("Unet48 (E)", "orange", "*", "solid"),
}
model_list = list(plot_mapping.keys())

# print table
output_types = ["M0.1 1-step", "M0.1 rollout", "M1.0 1-step", "M1.0 rollout"]
print("model ", end="")
for output_type in output_types:
    print(f"& {output_type} ", end="")

print("\\\\")
print("\\toprule")

block_size = 3  # number of models per block
for i in range(len(model_list) // block_size):
    for l in range(block_size):  # models come in a baseline and equiv small, and equiv large
        idx = block_size * i + l
        print(f"{plot_mapping[model_list[idx]][0]} ", end="")

        for mean_results, std_results in [(mean_M01, std_M01), (mean_M10, std_M10)]:
            for j in range(2, 4):  # only want the test and rollout test
                if jnp.trunc(std_results[0, idx, j] * 1000) / 1000 > 0:
                    stdev = f"$\\pm$ {std_results[0,idx,j]:.3f}"
                else:
                    stdev = ""

                if jnp.allclose(
                    mean_results[0, idx, j],
                    jnp.min(mean_results[0, block_size * i : block_size * (i + 1), j]),
                ):
                    print(f'& \\textbf{"{"}{mean_results[0,idx,j]:.3f} {stdev}{"}"}', end="")
                else:
                    print(f"& {mean_results[0,idx,j]:.3f} {stdev} ", end="")

        print("\\\\")

    if i < (len(model_list) // block_size) - 1:
        print("\\midrule")

print("\\bottomrule")

if args.images_dir:
    for idx, (model_name, (label, color, marker, linestyle)) in enumerate(plot_mapping.items()):
        plt.plot(
            jnp.arange(1, 1 + args.rollout_steps),
            jnp.mean(rollout_M01, axis=0)[0, idx],
            label=label,
            marker=marker,
            linestyle=linestyle,
            color=color,
        )

    plt.legend()
    plt.title(f"MSE vs. Rollout Step, Mean of {args.n_trials} Trials")
    plt.xlabel("Rollout Step")
    plt.ylabel("SMSE")
    plt.yscale("log")
    plt.savefig(f"{args.images_dir}/rollout_loss_plot.png")
    plt.close()
