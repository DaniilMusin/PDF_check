"""
Тесты для модуля модели машинного обучения.
"""
import sys
from pathlib import Path

import pytest
import numpy as np

# Добавляем src в путь
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from model.train import prepare_data, create_model
from model.predict import classify, translate_category


class TestModelTraining:
    """Тесты обучения модели."""
    
    def test_prepare_data(self):
        """Тест подготовки данных."""
        X, y, labels, vectorizer = prepare_data()
        
        assert X.shape[0] > 0, "Должны быть данные"
        assert X.shape[1] > 0, "Должны быть признаки"
        assert len(y) == X.shape[0], "Количество меток должно соответствовать количеству образцов"
        assert len(labels) == 20, "Должно быть 20 категорий Newsgroups"
        assert vectorizer is not None, "Векторизатор должен быть создан"
    
    def test_create_model(self):
        """Тест создания модели."""
        model = create_model(input_dim=1000, num_classes=20)
        
        assert model is not None, "Модель должна быть создана"
        assert len(model.layers) > 0, "Модель должна иметь слои"
        assert model.input_shape[1] == 1000, "Входная размерность должна соответствовать"
        assert model.output_shape[1] == 20, "Выходная размерность должна соответствовать"


class TestModelPrediction:
    """Тесты предсказания модели."""
    
    def test_translate_category(self):
        """Тест перевода категорий."""
        assert translate_category('comp.graphics') == 'Компьютерная графика'
        assert translate_category('sci.med') == 'Медицина'
        assert translate_category('unknown') == 'unknown'  # Неизвестная категория
    
    @pytest.mark.skipif(
        not Path("artifacts/model.keras").exists(),
        reason="Модель не обучена"
    )
    def test_classify_with_model(self, sample_pdf):
        """Тест классификации с обученной моделью."""
        if sample_pdf.exists():
            results = classify(sample_pdf, top_k=3)
            
            assert len(results) <= 3, "Должно быть не больше 3 результатов"
            assert all(isinstance(r, tuple) and len(r) == 2 for r in results), "Результат должен быть списком кортежей"
            assert all(isinstance(r[1], float) for r in results), "Вероятности должны быть числами"
            assert all(0 <= r[1] <= 1 for r in results), "Вероятности должны быть от 0 до 1"


@pytest.fixture
def sample_pdf(tmp_path):
    """Создает тестовый PDF файл."""
    pdf_path = tmp_path / "test.pdf"
    
    # Создаем минимальный PDF
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