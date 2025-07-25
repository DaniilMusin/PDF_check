from unittest import mock

import numpy as np

from model.predict import classify


def test_classify(monkeypatch, tmp_path):
    def fake_extract_text(path):
        return "hello"

    class DummyModel:
        def predict(self, arr, verbose=0):
            return np.array([[0.1, 0.9]])

    monkeypatch.setattr("model.predict.extract_text", fake_extract_text)
    monkeypatch.setattr("model.predict.load_model_and_vectorizer", lambda: (DummyModel(), lambda x: x, ["a", "b"]))
    res = classify("dummy.pdf")
    assert res[0][0] == "b"
