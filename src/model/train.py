"""
Обучение нейронной сети для классификации текста
(20\u00a0Newsgroups \u2192 7\u00a0суперкатегорий).

Скрипт можно запускать как модуль:
    $ python -m src.model.train --config configs/default.yaml

По-умолчанию используется встроенная конфигурация (см. ниже).
Веса модели и Vectorizer сохраняются в директорию *artifacts/*.
"""
from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from textwrap import shorten
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.datasets import fetch_20newsgroups
from sklearn.model_selection import train_test_split
from tensorflow.keras import callbacks, layers as L, models

from preprocessing.cleaner import preprocess

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
DEFAULT_CONFIG: dict[str, Any] = {
    "max_tokens": 20_000,
    "sequence_len": 300,
    "embedding_dim": 128,
    "rnn_units": 64,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-3,
    "patience": 3,
    "val_split": 0.15,
    "test_split": 0.15,
    "seed": 42,
    "super_categories": {
        # newsgroup : super-class
        "comp.graphics": "tech",
        "comp.sys.ibm.pc.hardware": "tech",
        "comp.sys.mac.hardware": "tech",
        "comp.windows.x": "tech",
        "comp.os.ms-windows.misc": "tech",
        "sci.crypt": "sci",
        "sci.electronics": "sci",
        "sci.med": "sci",
        "sci.space": "sci",
        "rec.autos": "autos",
        "rec.motorcycles": "autos",
        "rec.sport.baseball": "sport",
        "rec.sport.hockey": "sport",
        "talk.politics.guns": "politics",
        "talk.politics.mideast": "politics",
        "talk.politics.misc": "politics",
        "talk.religion.misc": "religion",
        "alt.atheism": "religion",
        "soc.religion.christian": "religion",
        "misc.forsale": "sale",
    },
}

ARTIFACTS = Path("artifacts")
ARTIFACTS.mkdir(exist_ok=True)


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train PDF text classifier.")
    p.add_argument(
        "--config",
        type=Path,
        help="Path to JSON/YAML config overriding defaults.",
    )
    return p.parse_args()


def _load_config(path: Path | None) -> dict[str, Any]:
    """Merge user config over defaults (supports JSON and YAML)."""
    if path is None:
        return DEFAULT_CONFIG
    if path.suffix.lower() in {".json"}:
        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = json.load(fh)
    elif path.suffix.lower() in {".yaml", ".yml"}:
        import yaml  # lazy import to keep dependency optional

        with open(path, "r", encoding="utf-8") as fh:
            user_cfg = yaml.safe_load(fh)
    else:
        raise ValueError("Config must be .json or .yaml")
    # recursive merge (shallow is enough here)
    merged = DEFAULT_CONFIG | user_cfg
    return merged


# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

def _group_labels(orig_targets: list[int], names: list[str], mapping: dict[str, str]):
    """Collapse 20 classes \u2192 6-7\u00a0суперкатегорий."""
    new_targets = [mapping[names[i]] for i in orig_targets]
    unique = sorted(set(new_targets))
    idx = {name: i for i, name in enumerate(unique)}
    return np.array([idx[t] for t in new_targets]), unique


def _prepare_dataset(cfg: dict[str, Any]):
    logger.info("Fetching 20\u00a0Newsgroups…")
    data = fetch_20newsgroups(remove=("headers", "footers", "quotes"))
    y, labels = _group_labels(data.target.tolist(), data.target_names, cfg["super_categories"])
    logger.info("Classes: %s", Counter(y))
    # Preprocess \u2192  space-joined tokens
    X = [" ".join(preprocess(t)) for t in data.data]
    test_size = cfg["test_split"] + cfg["val_split"]
    X_train, X_tmp, y_train, y_tmp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=cfg["seed"],
        stratify=y,
    )
    val_size = cfg["val_split"] / test_size
    X_val, X_test, y_val, y_test = train_test_split(
        X_tmp,
        y_tmp,
        test_size=val_size,
        random_state=cfg["seed"],
        stratify=y_tmp,
    )
    return (X_train, y_train), (X_val, y_val), (X_test, y_test), labels


def _build_model(cfg: dict[str, Any], num_classes: int, vectorizer: tf.keras.layers.TextVectorization):
    model = models.Sequential(
        [
            vectorizer,
            L.Embedding(cfg["max_tokens"], cfg["embedding_dim"]),
            L.Bidirectional(L.LSTM(cfg["rnn_units"], dropout=0.2)),
            L.Dense(num_classes, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(cfg["learning_rate"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main() -> None:
    args = _parse_args()
    cfg = _load_config(args.config)
    logger.info("Config: %s", cfg)

    (X_train, y_train), (X_val, y_val), (X_test, y_test), labels = _prepare_dataset(cfg)

    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=cfg["max_tokens"],
        output_sequence_length=cfg["sequence_len"],
        standardize=None,  # уже пред-обработано
    )
    vectorizer.adapt(X_train)

    model = _build_model(cfg, len(labels), vectorizer)

    logger.info("Starting training: %d samples, %d classes", len(X_train), len(labels))
    history = model.fit(
        X_train,
        y_train,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        validation_data=(X_val, y_val),
        callbacks=[
            callbacks.EarlyStopping(
                patience=cfg["patience"], restore_best_weights=True, verbose=1
            ),
        ],
        verbose=2,
    )

    logger.info(
        "Best val_acc: %.3f",
        max(history.history["val_accuracy"]),
    )
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    logger.success("Test accuracy: %.3f  •  loss: %.3f", test_acc, test_loss)

    # \u2014— Сохранение ——————————————————————————————————
    model_path = ARTIFACTS / "model.keras"
    model.save(model_path)
    with open(ARTIFACTS / "vectorizer.pkl", "wb") as fh:
        pickle.dump(vectorizer, fh)
    with open(ARTIFACTS / "labels.pkl", "wb") as fh:
        pickle.dump(labels, fh)
    logger.info("Artifacts saved to %s", ARTIFACTS.resolve())


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user")
