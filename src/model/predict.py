"""
PDF \u2192 \u043f\u0440\u0435\u0434\u0441\u043a\u0430\u0437\u0430\u043d\u0438\u0435 \u043a\u0430\u0442\u0435\u0433\u043e\u0440\u0438\u0438 (Top\u2011K).

\u042d\u043a\u0441\u043f\u043e\u0440\u0442\u0438\u0440\u0443\u0435\u043c\u0430\u044f \u0444\u0443\u043d\u043a\u0446\u0438\u044f:
    classify(pdf_path: str | Path, top_k: int = 1)
\u0432\u043e\u0437\u0432\u0440\u0430\u0449\u0430\u0435\u0442 \u0441\u043f\u0438\u0441\u043e\u043a \u043a\u043e\u0440\u0442\u0435\u0436\u0435\u0439 (\u043c\u0435\u0442\u043a\u0430, \u0432\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0441\u0442\u044c\u00a00\u20111).
"""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
from loguru import logger

from pdf_io.extractor import extract_text
from preprocessing.cleaner import preprocess
from model.load import load_model_and_vectorizer

__all__ = ["classify"]


def _prepare_input(texts: Iterable[str], vectorizer):
    """\u041f\u0440\u0435\u043e\u0431\u0440\u0430\u0437\u0443\u0435\u0442 \u0441\u043f\u0438\u0441\u043e\u043a \u0441\u0442\u0440\u043e\u043a \u2192 Tensor\u00a0\u0434\u043b\u044f \u043c\u043e\u0434\u0435\u043b\u0438."""
    return vectorizer(list(texts))


def classify(pdf_path: str | Path, top_k: int = 1) -> list[tuple[str, float]]:
    """
    \u041a\u043b\u0430\u0441\u0441\u0438\u0444\u0438\u0446\u0438\u0440\u0443\u0435\u0442 PDF\u2011\u0434\u043e\u043a\u0443\u043c\u0435\u043d\u0442.

    Parameters
    ----------
    pdf_path :
        \u041f\u0443\u0442\u044c \u043a \u0444\u0430\u0439\u043b\u0443.
    top_k :
        \u0421\u043a\u043e\u043b\u044c\u043a\u043e \u043b\u0443\u0447\u0448\u0438\u0445 \u043a\u043b\u0430\u0441\u0441\u043e\u0432 \u0432\u0435\u0440\u043d\u0443\u0442\u044c.

    Returns
    -------
    list[tuple[label, probability]]
        \u0421\u043e\u0440\u0442\u0438\u0440\u043e\u0432\u043a\u0430 \u043f\u043e \u0443\u0431\u044b\u0432\u0430\u043d\u0438\u044e \u0432\u0435\u0440\u043e\u044f\u0442\u043d\u043e\u0441\u0442\u0438.
    """
    pdf_path = Path(pdf_path)
    logger.info("Classifying %s\u2026", pdf_path.name)

    model, vectorizer, labels = load_model_and_vectorizer()

    raw_text = extract_text(pdf_path)
    tokens = " ".join(preprocess(raw_text))
    inp = _prepare_input([tokens], vectorizer)

    probs: np.ndarray = model.predict(inp, verbose=0)[0]  # shape == (n_classes,)
    top_idx = probs.argsort()[-top_k:][::-1]
    results = [(labels[i], float(probs[i])) for i in top_idx]
    logger.debug("Prediction: %s", results)
    return results
