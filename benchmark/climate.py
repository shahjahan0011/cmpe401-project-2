"""Reproduce the Jena Climate LSTM and evaluate its best validation checkpoint."""

import argparse
from pathlib import Path
from time import perf_counter

from .common import download, environment, prepare, provenance, write_json

import keras
import numpy as np
import pandas as pd
import tensorflow as tf

URL = (
    "https://storage.googleapis.com/tensorflow/tf-keras-datasets/"
    "jena_climate_2009_2016.csv.zip"
)
FEATURES = (
    "p (mbar)", "T (degC)", "VPmax (mbar)", "VPdef (mbar)",
    "sh (g/kg)", "rho (g/m**3)", "wv (m/s)",
)
PAST = 720
FUTURE = 72
STEP = 6
LENGTH = PAST // STEP
DATE_COLUMN = "Date Time"
DATE_FORMAT = "%d.%m.%Y %H:%M:%S"


def windows(inputs, targets, batch_size):
    """Match Keras's window order and hourly input sampling."""
    dataset = keras.utils.timeseries_dataset_from_array(
        inputs, targets, sequence_length=LENGTH, sampling_rate=STEP,
        batch_size=batch_size,
    )
    options = tf.data.Options()
    options.threading.private_threadpool_size = 4
    return dataset.with_options(options)


def datasets(raw, batch_size):
    split = int(0.715 * len(raw))
    mean = raw[:split].mean(axis=0)
    std = raw[:split].std(axis=0)
    features = ((raw - mean) / std).astype(np.float32)
    offset = PAST + FUTURE
    train = windows(features[:split], features[offset:offset + split, 1:2], batch_size)
    val = windows(features[split:-offset], features[split + offset:, 1:2], batch_size)
    span = (LENGTH - 1) * STEP
    counts = {
        "rows": len(raw),
        "split_row": split,
        "training_windows": split - span,
        "validation_windows": len(raw) - split - offset - span,
        "training_targets_at_or_after_split": offset - span,
        "raw_steps_from_last_input_to_target": offset - span,
        "unused_validation_rows_at_end": span,
    }
    return train, val, mean, std, counts


def target_timestamps(frame, counts):
    start = counts["split_row"] + PAST + FUTURE
    return pd.to_datetime(
        frame[DATE_COLUMN].iloc[start:start + counts["validation_windows"]],
        format=DATE_FORMAT,
    ).reset_index(drop=True)


def evaluate(model, dataset, timestamps, temperature_mean, temperature_std, output):
    observations, predictions, last_values = [], [], []
    for inputs, labels in dataset:
        observations.append(labels.numpy().ravel())
        predictions.append(model(inputs, training=False).numpy().ravel())
        last_values.append(inputs[:, -1, 1].numpy())
    y = np.concatenate(observations).astype(np.float64)
    predicted = np.concatenate(predictions).astype(np.float64)
    persistence = np.concatenate(last_values).astype(np.float64)
    records = pd.DataFrame({
        "target_timestamp": timestamps,
        "observed_c": y * temperature_std + temperature_mean,
        "lstm_c": predicted * temperature_std + temperature_mean,
        "persistence_c": persistence * temperature_std + temperature_mean,
    })
    records.to_csv(output / "predictions.csv.gz", index=False, compression={
        "method": "gzip", "mtime": 0,
    })
    metrics = {}
    for name, estimate in [("lstm", predicted), ("persistence", persistence)]:
        residual = estimate - y
        mse = float(np.mean(residual ** 2))
        metrics[name] = {
            "mse_normalized": mse,
            "mae_c": float(np.abs(residual).mean() * temperature_std),
            "rmse_c": float(np.sqrt(mse) * temperature_std),
        }
    return metrics


def build_model():
    inputs = keras.Input(shape=(LENGTH, len(FEATURES)))
    predicted = keras.layers.Dense(1)(keras.layers.LSTM(32)(inputs))
    return keras.Model(inputs, predicted)


def run(args):
    prepare(args.seed)
    path = download(args.data, URL)
    frame = pd.read_csv(path, compression="zip")
    raw = frame.loc[:, list(FEATURES)].to_numpy()
    train, val, mean, std, counts = datasets(raw, args.batch_size)
    model = build_model()
    model.compile(optimizer=keras.optimizers.Adam(0.001), loss="mse")

    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    history = model.fit(
        train, epochs=args.epochs, validation_data=val, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=5),
            keras.callbacks.ModelCheckpoint(
                output / "model.weights.h5", monitor="val_loss",
                save_best_only=True, save_weights_only=True,
            ),
            keras.callbacks.CSVLogger(output / "history.csv"),
        ],
    ).history
    training_seconds = perf_counter() - start
    model.load_weights(output / "model.weights.h5")
    timestamps = target_timestamps(frame, counts)
    metrics = evaluate(model, val, timestamps, mean[1], std[1], output)
    assert abs(metrics["lstm"]["mse_normalized"] - min(history["val_loss"])) < 1e-4
    result = {
        "seed": args.seed, "epoch_limit": args.epochs, "batch_size": args.batch_size,
        "epochs_run": len(history["loss"]),
        "best_epoch": int(np.argmin(history["val_loss"])) + 1,
        "parameters": model.count_params(), "training_seconds": training_seconds,
        "dataset": counts, "normalization": {"mean": mean.tolist(), "std": std.tolist()},
        "metrics": metrics, "dataset_sha256": provenance(path),
        "environment": environment(),
    }
    write_json(output / "metrics.json", result)
    print(metrics, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/climate"))
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=10)
    run(parser.parse_args())


if __name__ == "__main__":
    main()
