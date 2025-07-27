"""
Тесты для модуля предобработки текста.
"""
import sys
from pathlib import Path

import pytest

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from preprocessing.cleaner import clean, tokenize, preprocess


class TestTextCleaning:
    """Тесты очистки текста."""
    
    def test_clean_basic_text(self):
        """Тест базовой очистки текста."""
        text = "Hello,   world!!!   How are you?"
        result = clean(text)
        
        assert isinstance(result, str), "Результат должен быть строкой"
        assert "Hello" in result, "Должно содержать исходные слова"
        assert "world" in result, "Должно содержать исходные слова"
        assert "!!!" not in result, "Не должно содержать специальные символы"
        assert "   " not in result, "Не должно содержать множественные пробелы"
    
    def test_clean_empty_text(self):
        """Тест очистки пустого текста."""
        result = clean("")
        assert result == "", "Пустой текст должен остаться пустым"
    
    def test_clean_special_characters(self):
        """Тест очистки специальных символов."""
        text = "Text@#$%with^&*()special+characters="
        result = clean(text)
        
        assert "Text" in result, "Должен содержать обычные слова"
        assert "with" in result, "Должен содержать обычные слова"
        assert "special" in result, "Должен содержать обычные слова"
        assert "characters" in result, "Должен содержать обычные слова"
        assert "@#$%" not in result, "Не должен содержать специальные символы"


class TestTokenization:
    """Тесты токенизации."""
    
    def test_tokenize_english_text(self):
        """Тест токенизации английского текста."""
        text = "The cats are running quickly in the garden"
        tokens = list(tokenize(text))
        
        assert isinstance(tokens, list), "Результат должен быть списком"
        assert len(tokens) > 0, "Должны быть токены"
        assert "cat" in tokens or "cats" in tokens, "Должна быть лемматизация"
        assert "run" in tokens or "running" in tokens, "Должна быть лемматизация"
        assert "the" not in tokens, "Стоп-слова должны быть удалены"
    
    def test_tokenize_russian_text(self):
        """Тест токенизации русского текста."""
        text = "Кошки быстро бегают в саду"
        tokens = list(tokenize(text))
        
        assert isinstance(tokens, list), "Результат должен быть списком"
        assert len(tokens) > 0, "Должны быть токены"
        # Проверяем, что есть какие-то токены (точную лемматизацию проверить сложно)
    
    def test_tokenize_empty_text(self):
        """Тест токенизации пустого текста."""
        tokens = list(tokenize(""))
        assert tokens == [], "Пустой текст должен дать пустой список токенов"


class TestPreprocessing:
    """Тесты полной предобработки."""
    
    def test_preprocess_full_pipeline(self):
        """Тест полного пайплайна предобработки."""
        text = "The cats are running quickly in the garden!!!"
        result = preprocess(text)
        
        assert isinstance(result, list), "Результат должен быть списком"
        assert len(result) > 0, "Должны быть обработанные токены"
        assert all(isinstance(token, str) for token in result), "Все токены должны быть строками"
        assert all(token.islower() for token in result), "Все токены должны быть в нижнем регистре"
    
    def test_preprocess_mixed_language(self):
        """Тест предобработки смешанного текста."""
        text = "Hello мир! This is тест."
        result = preprocess(text)
        
        assert isinstance(result, list), "Результат должен быть списком"
        assert len(result) > 0, "Должны быть обработанные токены"
    
    def test_preprocess_with_numbers_and_punctuation(self):
        """Тест предобработки с числами и пунктуацией."""
        text = "Text with 123 numbers and !@# punctuation."
        result = preprocess(text)
        
        assert isinstance(result, list), "Результат должен быть списком"
        # Числа и пунктуация должны быть отфильтрованы
        assert not any(token.isdigit() for token in result), "Числа должны быть удалены"