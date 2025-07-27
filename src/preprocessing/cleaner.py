"""
Предобработка текста: очистка, токенизация, лемматизация
(русский + английский).

Функции нижнего уровня (`clean`, `tokenize`) имеют минимальные
зависимости и легко тестируются.  Высокоуровневая функция
`preprocess` возвращает список строк, готовый для подачи в
TextVectorization / CountVectorizer и т.д.
"""
from __future__ import annotations

import re
from functools import lru_cache
from typing import Iterable, Iterator

import spacy
from loguru import logger

__all__ = ["clean", "tokenize", "preprocess"]

# ────────────────────────────────────────────────────────────────────────────
# Регулярные выражения для грубой очистки
_RE_SPECIAL = re.compile(r"[^A-Za-zА-Яа-я0-9\s]")
_RE_SPACES = re.compile(r"\s+")


def clean(text: str) -> str:
    """
    Удаляет спец-символы и приводит множественные пробелы
    к одному.

    >>> clean("Hello,   world!!!")
    'Hello world'
    """
    text = _RE_SPECIAL.sub(" ", text)
    return _RE_SPACES.sub(" ", text).strip()


# ────────────────────────────────────────────────────────────────────────────
# spaCy — лениво грузим модели (дорого ~100 МБ каждая)


@lru_cache(maxsize=1)
def _nlp_en():
    return spacy.load("en_core_web_sm", disable=["parser", "ner", "textcat"])


@lru_cache(maxsize=1)
def _nlp_ru():
    try:
        return spacy.load("ru_core_news_sm", disable=["parser", "ner", "textcat"])
    except OSError:
        logger.warning("Russian spaCy model not found, falling back to English model")
        return _nlp_en()


def _select_nlp(text: str) -> spacy.language.Language:
    """Выбираем модель по наличию кириллицы."""
    if any("а" <= ch.lower() <= "я" for ch in text):
        return _nlp_ru()
    return _nlp_en()


def tokenize(text: str) -> Iterator[str]:
    """
    Генерирует токены: лемматизированные, в нижнем регистре,
    без стоп-слов, пунктуации и чисел.

    Parameters
    ----------
    text : str
        *Уже очищенный* текст.

    Yields
    ------
    str
        Очередной токен.
    """
    nlp = _select_nlp(text)
    doc = nlp(text)
    for token in doc:
        if token.is_stop or token.is_punct or token.like_num:
            continue
        yield token.lemma_.lower()


def preprocess(text: str) -> list[str]:
    """
    Полный конвейер: raw text → clean → tokenize → list[str].

    >>> preprocess("Cats, cats, CATS!!! 123")
    ['cat']
    """
    logger.debug("Preprocessing text (%d chars)", len(text))
    cleaned = clean(text)
    return list(tokenize(cleaned))
