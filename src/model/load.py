"""
Утилиты для ленивой загрузки обученной модели и Vectorizer'а.
Функция кэшируется через @lru_cache — поэтому модель
загружается с диска только один раз на процесс.
"""
from __future__ import annotations

import pickle
from functools import lru_cache
from pathlib import Path
from typing import Any, Tuple, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - used for type hints only
    import tensorflow as tf
from loguru import logger

__all__ = ["load_model_and_vectorizer"]

ARTIFACTS = Path("artifacts")
MODEL_FILE = ARTIFACTS / "model.keras"
VECT_FILE = ARTIFACTS / "vectorizer.pkl"
LABELS_FILE = ARTIFACTS / "labels.pkl"


@lru_cache(maxsize=1)
def load_model_and_vectorizer() -> Tuple["tf.keras.Model", Any, list[str]]:
    """
    Загружает сохранённые артефакты и кэширует их.

    Returns
    -------
    model : tf.keras.Model
    vectorizer : tf.keras.layers.TextVectorization
    labels : list[str]
    """
    import tensorflow as tf
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}. Run training first."
        )

    logger.debug("Loading model from %s…", MODEL_FILE.name)
    model = tf.keras.models.load_model(MODEL_FILE)

    with open(VECT_FILE, "rb") as fh:
        vectorizer = pickle.load(fh)
    with open(LABELS_FILE, "rb") as fh:
        labels = pickle.load(fh)

    logger.debug("Artifacts loaded: model, vectorizer, %d labels", len(labels))
    return model, vectorizer, labels
