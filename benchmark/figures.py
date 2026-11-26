"""Draw benchmark figures from the saved predictions and epoch logs."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def style_axes(axis):
    axis.grid(axis="y", color="#dfe6ed", linewidth=0.8)
    axis.set_axisbelow(True)
    axis.spines[["right", "top"]].set_visible(False)


def forda(source, destination):
    colors = ("#6b7280", "#007c91", "#a15012", "#4147a5")
    names = {
        "baseline": "Current Keras baseline",
        "samplewise": "Keep all sample positions",
        "patch": "Patch embedding",
        "patch-position": "Patch embedding + position",
    }
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for (config, label), color in zip(names.items(), colors):
        history = pd.read_csv(source / config / "history.csv")
        epochs = history["epoch"] + 1
        axes[0].plot(epochs, history["val_loss"], label=label, color=color, linewidth=2)
        axes[1].plot(
            epochs, history["val_sparse_categorical_accuracy"] * 100,
            label=label, color=color, linewidth=2,
        )
        best = history["val_loss"].idxmin()
        axes[0].scatter(epochs[best], history.loc[best, "val_loss"], color=color, s=18)
        axes[1].scatter(
            epochs[best], history.loc[best, "val_sparse_categorical_accuracy"] * 100,
            color=color, s=18,
        )
    axes[0].set(title="Validation cross-entropy", xlabel="Epoch", ylabel="Loss")
    axes[1].set(title="Validation classification accuracy", xlabel="Epoch", ylabel="Accuracy (%)")
    axes[0].legend(loc="best", frameon=False, fontsize=8)
    for axis in axes:
        style_axes(axis)
    figure.savefig(destination / "forda-validation.png", dpi=180)
    plt.close(figure)


def climate(source, destination):
    history = pd.read_csv(source / "climate" / "history.csv")
    forecast = pd.read_csv(
        source / "climate" / "predictions.csv.gz", parse_dates=["target_timestamp"],
    )
    figure, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    epoch = history["epoch"] + 1
    axes[0].plot(epoch, history["loss"], label="Train", color="#007c91", marker="o")
    axes[0].plot(epoch, history["val_loss"], label="Validation", color="#4147a5", marker="o")
    axes[0].set(title="LSTM training and validation loss", xlabel="Epoch", ylabel="Normalized MSE")
    axes[0].set_xticks(epoch)
    # Plot every sixth target on the original timestamp axis and break the lines at gaps.
    excerpt = forecast.iloc[:432:6].copy()
    time = excerpt["target_timestamp"]
    excerpt["elapsed_hours"] = (time - time.iloc[0]).dt.total_seconds() / 3600
    segment = time.diff().ne(pd.Timedelta(hours=1)).cumsum()
    for index, plot in excerpt.groupby(segment):
        hours = plot["elapsed_hours"]
        first = index == segment.iloc[0]
        axes[1].plot(hours, plot["observed_c"], color="#172735", linewidth=2,
                     label="Observed" if first else None)
        axes[1].plot(hours, plot["lstm_c"], color="#007c91", linestyle="--",
                     label="LSTM" if first else None)
        axes[1].plot(hours, plot["persistence_c"], color="#a15012", linestyle=":",
                     label="Persistence" if first else None)
    axes[1].set(
        title="Validation forecast excerpt",
        xlabel="Elapsed hours from first target",
        ylabel="Temperature (°C)",
    )
    for axis in axes:
        style_axes(axis)
        axis.legend(frameon=False)
    figure.savefig(destination / "jena-forecast.png", dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/figures"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    forda(args.source, args.output)
    climate(args.source, args.output)


if __name__ == "__main__":
    main()
