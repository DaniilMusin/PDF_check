from preprocessing.cleaner import clean, preprocess


def test_clean():
    assert clean("Hello!!!   Мир???") == "Hello Мир"


def test_preprocess():
    tokens = preprocess("Hello world")
    assert "hello" in tokens
