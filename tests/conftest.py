"""
Конфигурация pytest и общие фикстуры для тестов.
"""
import pytest
from pathlib import Path


@pytest.fixture(scope="session")
def test_data_dir():
    """Путь к директории с тестовыми данными."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_text():
    """Пример текста для тестирования."""
    return """
    Computer graphics is the field of computer science that studies methods for 
    digitally synthesizing and manipulating visual content. Although the term 
    often refers to 3D computer graphics, it also encompasses 2D graphics and 
    image processing.
    """


@pytest.fixture
def russian_text():
    """Пример русского текста для тестирования."""
    return """
    Компьютерная графика — область информатики, изучающая методы цифрового 
    синтеза и обработки визуального контента. Она включает в себя как 
    двумерную, так и трёхмерную графику, а также обработку изображений.
    """