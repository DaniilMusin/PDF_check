"""
Модуль отвечает за извлечение текста из PDF-документов
с использованием каскада из трёх бэкендов:

1. **PyMuPDF**  – самый быстрый и надёжный (Unicode, графика, шифр).
2. **pdfminer.six** – точный low-level парсер (fallback #1).
3. **PyPDF2**   – резервный вариант (fallback #2).

Если все три варианта потерпели неудачу, поднимается
исключение :class:`PDFExtractionError`.

Пример использования
--------------------
```python
from pdf_io.extractor import extract_text

text = extract_text("document.pdf")
````
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Final

import fitz  # type: ignore[import]  # PyMuPDF
from loguru import logger
from pdfminer.high_level import extract_text as pdfminer_extract  # type: ignore[import]
from PyPDF2 import PdfReader  # type: ignore[import]

__all__ = ["extract_text", "PDFExtractionError"]


class PDFExtractionError(RuntimeError):
    """\u0411\u0440\u043e\u0441\u0430\u0435\u0442\u0441\u044f, \u0435\u0441\u043b\u0438 \u0442\u0435\u043a\u0441\u0442 \u043d\u0435 \u0443\u0434\u0430\u043b\u043e\u0441\u044c \u0438\u0437\u0432\u043b\u0435\u0447\u044c \u043d\u0438 \u043e\u0434\u043d\u0438\u043c \u0441\u043f\u043e\u0441\u043e\u0431\u043e\u043c."""


def _extract_mupdf(path: Path) -> str:
    """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0447\u0435\u0440\u0435\u0437 PyMuPDF (fitz)."""
    try:
        with fitz.open(path) as doc:
            return "\n".join(page.get_text() for page in doc)
    except (fitz.FileDataError, RuntimeError) as exc:  # noqa: BLE001
        raise exc


def _extract_pdfminer(path: Path) -> str:
    """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0447\u0435\u0440\u0435\u0437 pdfminer.six."""
    return pdfminer_extract(str(path))


def _extract_pypdf2(path: Path) -> str:
    """\u0418\u0437\u0432\u043b\u0435\u0447\u0435\u043d\u0438\u0435 \u0447\u0435\u0440\u0435\u0437 PyPDF2."""
    reader = PdfReader(path)
    return "\n".join(page.extract_text() or "" for page in reader.pages)


#: Очерёдность бэкендов — от самого быстрого к резервному.
_BACKENDS: Final[list[Callable[[Path], str]]] = [
    _extract_mupdf,
    _extract_pdfminer,
    _extract_pypdf2,
]


def extract_text(path: str | os.PathLike[str]) -> str:
    """
    Возвращает *сырой* текст из PDF-файла.

    ```
    Parameters
    ----------
    path :
        Путь к файлу.

    Returns
    -------
    str
        Извлечённый текст без дополнительной обработки.

    Raises
    ------
    PDFExtractionError
        Если все бэкенды завершились неудачей.
    """
    pdf_path = Path(path)
    if not pdf_path.is_file():
        raise FileNotFoundError(pdf_path)

    for backend in _BACKENDS:
        backend_name = backend.__name__.lstrip("_")
        try:
            text = backend(pdf_path).strip()
            if text:
                logger.debug("Extractor %s succeeded for %s", backend_name, pdf_path.name)
                return text
            logger.warning("Extractor %s returned empty text for %s", backend_name, pdf_path.name)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Extractor %s failed (%s): %s", backend_name, pdf_path.name, exc)

    raise PDFExtractionError(f"Unable to extract text from {pdf_path!s}")
