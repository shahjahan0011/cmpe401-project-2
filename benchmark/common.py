"""Shared data provenance and reproducibility helpers."""

import hashlib
import json
import os
import platform
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve

os.environ.setdefault("KERAS_BACKEND", "tensorflow")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import keras
import tensorflow as tf


def prepare(seed):
    keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()


def download(directory, url):
    path = directory / url.rsplit("/", 1)[-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        with NamedTemporaryFile(dir=directory, suffix=".part", delete=False) as stream:
            partial = Path(stream.name)
        try:
            urlretrieve(url, partial)
            partial.replace(path)
        finally:
            partial.unlink(missing_ok=True)
    return path


def provenance(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def environment():
    return {
        "python": platform.python_version(),
        "keras": keras.__version__,
        "tensorflow": tf.__version__,
        "platform": platform.platform(),
        "devices": [device.device_type for device in tf.config.list_physical_devices()],
        "intraop_threads": os.environ["TF_NUM_INTRAOP_THREADS"],
        "interop_threads": os.environ["TF_NUM_INTEROP_THREADS"],
    }


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
