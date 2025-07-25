"""
pdf_classifier — пакет для извлечения текста из PDF и
последующей тематической классификации документов.

Состав пакета:
    • pdf_io        – работа с PDF-файлами
    • preprocessing – очистка и токенизация текста
    • model         – обучение и инференс
    • app           – CLI и Flask-веб-интерфейс
"""
from __future__ import annotations

__all__ = [
    "pdf_io",
    "preprocessing",
    "model",
    "app",
]

__version__: str = "0.1.0"
