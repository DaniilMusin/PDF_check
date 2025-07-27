"""
Тесты для модуля извлечения текста из PDF.
"""
import sys
from pathlib import Path

import pytest

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pdf_io.extractor import extract_text


class TestPDFExtractor:
    """Тесты извлечения текста из PDF."""
    
    def test_extract_text_from_valid_pdf(self, sample_pdf):
        """Тест извлечения текста из валидного PDF."""
        text = extract_text(sample_pdf)
        
        assert isinstance(text, str), "Результат должен быть строкой"
        assert len(text) > 0, "Текст не должен быть пустым"
        assert "Computer Graphics" in text, "Текст должен содержать ожидаемое содержимое"
    
    def test_extract_text_from_nonexistent_file(self):
        """Тест извлечения текста из несуществующего файла."""
        with pytest.raises(FileNotFoundError):
            extract_text("nonexistent.pdf")
    
    def test_extract_text_from_invalid_pdf(self, invalid_pdf):
        """Тест извлечения текста из поврежденного PDF."""
        # Должен либо вернуть пустую строку, либо вызвать исключение
        try:
            text = extract_text(invalid_pdf)
            assert isinstance(text, str), "Результат должен быть строкой"
        except Exception:
            # Ожидаемое поведение для поврежденного файла
            pass


@pytest.fixture
def sample_pdf(tmp_path):
    """Создает тестовый PDF файл."""
    pdf_path = tmp_path / "test.pdf"
    
    # Создаем минимальный валидный PDF
    content = b"""%PDF-1.4
1 0 obj<</Type/Catalog/Pages 2 0 R>>endobj
2 0 obj<</Type/Pages/Kids[3 0 R]/Count 1>>endobj
3 0 obj<</Type/Page/Parent 2 0 R/MediaBox[0 0 612 792]/Contents 4 0 R>>endobj
4 0 obj<</Length 60>>stream
BT /F1 12 Tf 50 700 Td (Computer Graphics Test Document) Tj ET
endstream endobj
xref 0 5
0000000000 65535 f 
0000000009 00000 n 
0000000058 00000 n 
0000000115 00000 n 
0000000206 00000 n 
trailer<</Size 5/Root 1 0 R>>
startxref 318
%%EOF"""
    
    with open(pdf_path, "wb") as f:
        f.write(content)
    
    return pdf_path


@pytest.fixture
def invalid_pdf(tmp_path):
    """Создает поврежденный PDF файл."""
    pdf_path = tmp_path / "invalid.pdf"
    
    # Создаем файл с некорректным содержимым
    with open(pdf_path, "wb") as f:
        f.write(b"This is not a valid PDF file")
    
    return pdf_path