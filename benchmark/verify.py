"""Verify the recorded metrics, Jena window indices, and saved model outputs."""

import argparse
import json
from pathlib import Path

from . import climate, forda

import keras
import numpy as np
import pandas as pd


def read_run(directory):
    return json.loads((directory / "metrics.json").read_text()), pd.read_csv(directory / "history.csv")


def verify_forda(root, data):
    reported, _ = read_run(root / "baseline")
    _, _, test, hashes = forda.load_data(data, reported["seed"])
    # Zero-initialized positions leave the paired model predictions unchanged at initialization.
    forda.prepare(reported["seed"])
    patch = forda.build_model("patch")
    forda.prepare(reported["seed"])
    patch_position = forda.build_model("patch-position")
    np.testing.assert_array_equal(
        patch(test[0][:8], training=False).numpy(),
        patch_position(test[0][:8], training=False).numpy(),
    )
    for directory in sorted(root.iterdir()):
        file = directory / "predictions.csv"
        if not file.exists():
            continue
        metrics, history = read_run(directory)
        predictions = pd.read_csv(file)
        labels = predictions["label"].to_numpy()
        p1 = predictions["p_class_1"].to_numpy()
        predicted = (p1 > 0.5).astype(int)
        confusion = np.bincount(2 * labels + predicted, minlength=4).reshape(2, 2)
        np.testing.assert_array_equal(metrics["test_confusion_matrix"], confusion)
        np.testing.assert_array_equal(labels, test[1])
        np.testing.assert_allclose(metrics["test_accuracy"], np.mean(labels == predicted), atol=1e-7)
        np.testing.assert_allclose(metrics["test_balanced_accuracy"], (confusion.diagonal() / confusion.sum(axis=1)).mean())
        observed_probability = np.where(labels, p1, 1 - p1)
        np.testing.assert_allclose(metrics["test_loss"], -np.log(observed_probability).mean(), atol=1e-6)
        assert metrics["best_epoch"] == history["val_loss"].idxmin() + 1
        assert metrics["dataset_sha256"] == hashes

        model = keras.saving.load_model(directory / "model.keras", compile=False)
        n = 8
        original = model(test[0][:n], training=False).numpy()[:, 1]
        np.testing.assert_allclose(original, p1[:n], atol=2e-4)
        if metrics["config"] in {"patch", "patch-position"}:
            reversed_patches = test[0][:n].reshape(n, -1, forda.PATCH_SIZE)[:, ::-1, :].reshape(n, -1, 1)
            reordered = model(reversed_patches, training=False).numpy()[:, 1]
            difference = float(np.max(np.abs(original - reordered)))
            print(f"{directory.name} patch reversal max probability change: {difference:.8f}")
            if metrics["config"] == "patch":
                assert difference < 1e-5
            else:
                assert difference > 1e-5
        print(f"Verified {directory.name} ({len(labels)} test predictions)")


def verify_climate(root, data):
    directory = root / "climate"
    metrics, history = read_run(directory)
    path = climate.download(data, climate.URL)
    frame = pd.read_csv(path, compression="zip")
    raw = frame.loc[:, list(climate.FEATURES)].to_numpy()
    train, val, mean, std, counts = climate.datasets(raw, metrics["batch_size"])
    assert metrics["dataset_sha256"] == climate.provenance(path)
    assert metrics["dataset"] == counts
    assert metrics["best_epoch"] == history["val_loss"].idxmin() + 1
    assert (len(train) - 1) * metrics["batch_size"] < counts["training_windows"] <= len(train) * metrics["batch_size"]

    inputs, targets = next(iter(val))
    split = counts["split_row"]
    span = np.arange(climate.LENGTH) * climate.STEP
    np.testing.assert_allclose(inputs[0], (raw[split + span] - mean) / std, atol=1e-6)
    np.testing.assert_allclose(targets[0], (raw[split + climate.PAST + climate.FUTURE, 1] - mean[1]) / std[1], atol=1e-6)

    predictions = pd.read_csv(directory / "predictions.csv.gz", parse_dates=["target_timestamp"])
    pd.testing.assert_series_equal(
        predictions["target_timestamp"], climate.target_timestamps(frame, counts), check_names=False,
    )
    assert len(predictions) == counts["validation_windows"]
    target_start = split + climate.PAST + climate.FUTURE
    np.testing.assert_allclose(
        predictions["observed_c"], raw[target_start:target_start + len(predictions), 1], atol=1e-5,
    )
    for name in ("lstm", "persistence"):
        residual_c = predictions[f"{name}_c"] - predictions["observed_c"]
        errors = metrics["metrics"][name]
        np.testing.assert_allclose(errors["mse_normalized"], ((residual_c / std[1]) ** 2).mean())
        np.testing.assert_allclose(errors["mae_c"], residual_c.abs().mean())
        np.testing.assert_allclose(errors["rmse_c"], np.sqrt((residual_c ** 2).mean()))
    model = climate.build_model()
    model.load_weights(directory / "model.weights.h5")
    values = model(inputs, training=False).numpy().ravel() * std[1] + mean[1]
    np.testing.assert_allclose(values, predictions["lstm_c"][:len(inputs)], atol=1e-5)
    print(f"Verified climate ({len(predictions)} validation predictions and window alignment)")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--results", type=Path, default=Path("results"))
    args = parser.parse_args()
    forda.prepare(401)
    verify_forda(args.results, args.data)
    verify_climate(args.results, args.data)


if __name__ == "__main__":
    main()
