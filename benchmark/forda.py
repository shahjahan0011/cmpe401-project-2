"""Run the FordA Transformer baseline and three controlled modifications."""

import argparse
from pathlib import Path
from time import perf_counter

from .common import download, environment, prepare, provenance, write_json

import keras
import numpy as np
import pandas as pd

ROOT = "https://raw.githubusercontent.com/hfawaz/cd-diagram/master/FordA/"
CONFIGS = {
    "baseline": ("channels_last", False, False),
    "samplewise": ("channels_first", False, False),
    "patch": ("channels_last", True, False),
    "patch-position": ("channels_last", True, True),
}
SAMPLES = 500
PATCH_SIZE = 25
WIDTH = 16


@keras.saving.register_keras_serializable(package="forda")
class ScalarShift(keras.layers.Layer):
    """A residual plus the beta of a LayerNormalization on one channel."""

    def build(self, input_shape):
        self.beta = self.add_weight(name="beta", shape=(1,), initializer="zeros")

    def call(self, inputs):
        return inputs + self.beta


@keras.saving.register_keras_serializable(package="forda")
class PositionEmbedding(keras.layers.Layer):
    """Add a separate trainable vector at each patch position."""

    def build(self, input_shape):
        self.embedding = self.add_weight(
            name="embedding", shape=input_shape[1:],
            initializer="zeros",
        )

    def call(self, inputs):
        return inputs + self.embedding


def transformer_block(inputs, index, folded):
    norm = f"normalization_{index}"
    if folded:
        residual = ScalarShift(name=f"{norm}_attention")(inputs)
        return ScalarShift(name=f"{norm}_feedforward")(residual)

    layers = keras.layers
    attention = layers.MultiHeadAttention(
        key_dim=256, num_heads=4, dropout=0.25
    )(inputs, inputs)
    attention = layers.Dropout(0.25)(attention)
    attention = layers.LayerNormalization(
        epsilon=1e-6, name=f"{norm}_attention"
    )(attention)
    residual = layers.Add()([inputs, attention])
    hidden = layers.Conv1D(4, 1, activation="relu")(residual)
    hidden = layers.Dropout(0.25)(hidden)
    hidden = layers.Conv1D(inputs.shape[-1], 1)(hidden)
    hidden = layers.LayerNormalization(
        epsilon=1e-6, name=f"{norm}_feedforward"
    )(hidden)
    return layers.Add()([residual, hidden])


def build_model(config, folded=False):
    pooling, patch, position = CONFIGS[config]
    if folded and patch:
        raise ValueError("Only singleton-channel normalization can be folded.")
    layers = keras.layers
    inputs = keras.Input(shape=(SAMPLES, 1))
    hidden = inputs
    if patch:
        hidden = layers.Conv1D(WIDTH, PATCH_SIZE, strides=PATCH_SIZE)(hidden)
    if position:
        hidden = PositionEmbedding()(hidden)
    for index in range(4):
        hidden = transformer_block(hidden, index, folded)
    hidden = layers.GlobalAveragePooling1D(data_format=pooling)(hidden)
    hidden = layers.Dense(128, activation="relu", name="classifier_hidden")(hidden)
    hidden = layers.Dropout(0.4)(hidden)
    probabilities = layers.Dense(2, activation="softmax", name="classifier")(hidden)
    return keras.Model(inputs, probabilities)


def transfer(literal, folded, *, to_literal):
    for reduced_layer in folded.layers:
        if isinstance(reduced_layer, ScalarShift):
            full_layer = literal.get_layer(reduced_layer.name)
            if to_literal:
                full_layer.beta.assign(reduced_layer.beta)
            else:
                reduced_layer.beta.assign(full_layer.beta)
        elif isinstance(reduced_layer, keras.layers.Dense):
            full_layer = literal.get_layer(reduced_layer.name)
            source, destination = (
                (reduced_layer, full_layer) if to_literal else (full_layer, reduced_layer)
            )
            destination.set_weights(source.get_weights())


def load_data(directory, seed):
    train_path = download(directory, ROOT + "FordA_TRAIN.tsv")
    test_path = download(directory, ROOT + "FordA_TEST.tsv")
    train = np.loadtxt(train_path, delimiter="\t", dtype=np.float32)
    test = np.loadtxt(test_path, delimiter="\t", dtype=np.float32)
    train = train[np.random.RandomState(seed).permutation(len(train))]
    cut = int(0.8 * len(train))
    partitions = [(rows[:, 1:, None], (rows[:, 0] == 1).astype(np.int32))
                  for rows in (train[:cut], train[cut:], test)]
    return (*partitions, {path.name: provenance(path) for path in (train_path, test_path)})


def run(args):
    prepare(args.seed)
    train, val, test, hashes = load_data(args.data, args.seed)
    literal = build_model(args.config)
    folded = args.engine == "folded"
    model = build_model(args.config, folded=True) if folded else literal
    if folded:
        transfer(literal, model, to_literal=False)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["sparse_categorical_accuracy"],
    )
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    start = perf_counter()
    history = model.fit(
        *train, validation_data=val, epochs=args.epochs, batch_size=64, verbose=2,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.CSVLogger(output / "history.csv"),
        ],
    ).history
    training_seconds = perf_counter() - start

    equivalence = None
    if folded:
        transfer(literal, model, to_literal=True)
        probe = test[0][:16]
        literal_predictions = literal(probe, training=False).numpy()
        folded_predictions = model(probe, training=False).numpy()
        np.testing.assert_allclose(literal_predictions, folded_predictions, atol=2e-4)
        equivalence = float(np.max(np.abs(literal_predictions - folded_predictions)))
        literal.save(output / "model.keras")
    else:
        model.save(output / "model.keras")

    loss, accuracy = model.evaluate(*test, batch_size=64, verbose=0)
    probabilities = model.predict(test[0], batch_size=64, verbose=0)
    actual = test[1]
    predicted = probabilities.argmax(axis=1)
    confusion = np.bincount(2 * actual + predicted, minlength=4).reshape(2, 2)
    recalls = np.diag(confusion) / confusion.sum(axis=1)
    np.testing.assert_allclose(accuracy, np.mean(actual == predicted), atol=1e-6)
    pd.DataFrame({"label": actual, "p_class_1": probabilities[:, 1]}).to_csv(
        output / "predictions.csv", index=False,
    )
    best = int(np.argmin(history["val_loss"]))
    result = {
        "config": args.config, "engine": args.engine, "seed": args.seed,
        "epoch_limit": args.epochs, "epochs_run": len(history["loss"]),
        "best_epoch": best + 1, "val_loss": history["val_loss"][best],
        "val_accuracy": history["val_sparse_categorical_accuracy"][best],
        "test_loss": loss, "test_accuracy": accuracy,
        "test_balanced_accuracy": float(recalls.mean()),
        "test_confusion_matrix": confusion.tolist(),
        "majority_class": int(np.argmax(np.bincount(train[1]))),
        "majority_test_accuracy": float(np.mean(test[1] == np.argmax(np.bincount(train[1])))),
        "split_sizes": {"train": len(train[1]), "validation": len(val[1]), "test": len(actual)},
        "literal_parameters": literal.count_params(), "trained_graph_parameters": model.count_params(),
        "training_seconds": training_seconds,
        "folding_max_probability_delta": equivalence,
        "dataset_sha256": hashes, "environment": environment(),
    }
    write_json(output / "metrics.json", result)
    print(f"Test accuracy: {accuracy:.4f} | Test loss: {loss:.4f}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=CONFIGS, default="baseline")
    parser.add_argument("--engine", choices=("literal", "folded"), default="literal")
    parser.add_argument("--data", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("runs/forda"))
    parser.add_argument("--seed", type=int, default=401)
    parser.add_argument("--epochs", type=int, default=150)
    args = parser.parse_args()
    if args.engine == "folded" and CONFIGS[args.config][1]:
        parser.error("--engine folded only supports baseline and samplewise")
    run(args)


if __name__ == "__main__":
    main()
