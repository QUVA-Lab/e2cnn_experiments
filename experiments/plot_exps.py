
import pandas as pd
import argparse
import os
import matplotlib

import utils

if "DISPLAY" not in os.environ:
    matplotlib.use('Agg')

import matplotlib.pyplot as plt

SHOW_PLOT = False
SAVE_PLOT = True

RESHUFFLE = False
AUGMENT_TRAIN = False

colors = {
    "train": "blue",
    "valid": "green",
    "test": "red"
}


def plot_mean_with_variance(axis, data, label):
    mean = data.mean()
    std = data.std()
    axis.plot(mean, label=label, color=colors[label])
    axis.fill_between(
        mean.index,
        mean - std,
        mean + std,
        color=colors[label],
        alpha=0.1
    )


def plot(logs, plotpath=None, show=False, outfig=None):
    
    if isinstance(logs, str) and os.path.isfile(logs):
        logs = utils.retrieve_logs(logs)
    elif not isinstance(logs, pd.DataFrame):
        raise ValueError()
    
    if outfig is None:
        figure, (loss_axis, acc_axis) = plt.subplots(1, 2, figsize=(10, 4))
    else:
        figure, (loss_axis, acc_axis) = outfig

    train = logs[logs.split.str.startswith("train")].groupby("iteration")
    valid = logs[logs.split == "valid"].groupby("iteration")
    test = logs[logs.split == "test"].groupby("iteration")
    
    #################### Plot Loss trends ####################
    
    loss_axis.cla()
    
    plot_mean_with_variance(loss_axis, train.loss, "train")
    if len(valid) > 0:
        plot_mean_with_variance(loss_axis, valid.loss, "valid")
    if len(test) > 0:
        plot_mean_with_variance(loss_axis, test.loss, "test")
    
    loss_axis.legend()
    loss_axis.set_xlabel('iterations')
    loss_axis.set_ylabel('Loss')
    
    #################### Plot Accuracy trends ####################
    
    acc_axis.cla()
    
    plot_mean_with_variance(acc_axis, train.accuracy, "train")
    if len(valid) > 0:
        plot_mean_with_variance(acc_axis, valid.accuracy, "valid")
    if len(test) > 0:
        plot_mean_with_variance(acc_axis, test.accuracy, "test")
    
    ################## Test score ########################
    
    test = logs[logs.split == "test"]
    
    xmax = logs.iteration.max()
    
    if len(test) > 0:
        best_acc = test.accuracy.max()
        acc_axis.hlines(best_acc, xmin=0, xmax=xmax, linewidth=0.5, linestyles='--', label='Max Test Accuracy')
        acc_axis.set_yticks(list(acc_axis.get_yticks()) + [best_acc])
    
    if len(test) > 1:
        mean_acc = test.accuracy.mean()
        mean_std = test.accuracy.std()
        acc_axis.hlines(mean_acc, xmin=0, xmax=xmax, linewidth=0.5, color=colors["test"], label='Mean Test Accuracy')
        acc_axis.fill_between([0, xmax], [mean_acc - mean_std] * 2, [mean_acc + mean_std] * 2, color=colors["test"],
                              alpha=0.1)
        acc_axis.set_yticks(list(acc_axis.get_yticks()) + [mean_acc])
    
    acc_axis.legend()
    acc_axis.set_xlabel('iterations')
    acc_axis.set_ylabel('Accuracy')
    
    figure.tight_layout()
    plt.draw()
    
    if plotpath is not None:
        figure.savefig(plotpath, format='svg', dpi=256, bbox_inches="tight")
    
    if show:
        figure.show()
        plt.pause(0.01)


################################################################################
################################################################################


if __name__ == "__main__":
    # Parse training configuration
    parser = argparse.ArgumentParser()
    
    # Dataset params
    parser.add_argument('--dataset', type=str, help='The name of the dataset to use')
    parser.add_argument('--augment', dest="augment", action="store_true",
                        help='Augment the training set with rotated images')
    parser.set_defaults(augment=AUGMENT_TRAIN)

    parser.add_argument('--reshuffle', dest="reshuffle", action="store_true",
                        help='Reshuffle train and valid splits instead of using the default split')
    parser.set_defaults(reshuffle=RESHUFFLE)
    
    # Model params
    parser.add_argument('--model', type=str, help='The name of the model to use')
    parser.add_argument('--N', type=int, help='Size of cyclic group for GCNN and maximum frequency for HNET')
    parser.add_argument('--flip', dest="flip", action="store_true",
                        help='Use also reflection equivariance in the EXP model')
    parser.set_defaults(flip=False)
    parser.add_argument('--sgsize', type=int, default=None,
                        help='Number of rotations in the subgroup to restrict to in the EXP e2sfcnn models')
    parser.add_argument('--fixparams', dest="fixparams", action="store_true",
                        help='Keep the number of parameters of the model fixed by adjusting its topology')
    parser.set_defaults(fixparams=False)
    parser.add_argument('--F', type=float, default=0.8, help='Frequency cut-off: maximum frequency at radius "r" is "F*r"')
    parser.add_argument('--sigma', type=float, default=0.6, help='Width of the rings building the bases (std of the gaussian window)')
    parser.add_argument('--J', type=int, default=0, help='Number of additional frequencies in the interwiners of finite groups')
    parser.add_argument('--restrict', type=int, default=-1, help='Layer where to restrict SFCNN from E(2) to SE(2)')

    # plot configs
    parser.add_argument('--show', dest="show", action="store_true", help='Show the plots during execution')
    parser.set_defaults(show=SHOW_PLOT)

    parser.add_argument('--store_plot', dest="store_plot", action="store_true", help='Save the plots in a file or not')
    parser.set_defaults(store_plot=SAVE_PLOT)
    
    config = parser.parse_args()
    
    # Draw the plot
    logs_file = utils.logs_path(config)
    plotpath = utils.plot_path(config)
    plot(logs_file, plotpath, config.show)
