"""
Метрики для оценки качества модели классификации.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
from loguru import logger
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score
)
from sklearn.datasets import fetch_20newsgroups

from model.load import load_model_and_vectorizer
from preprocessing.cleaner import preprocess

__all__ = ["evaluate_model", "generate_report"]


def evaluate_model() -> dict[str, Any]:
    """
    Оценивает качество обученной модели на тестовых данных.
    
    Returns:
        Словарь с метриками качества
    """
    logger.info("Загрузка модели и тестовых данных...")
    
    # Загружаем модель
    model, vectorizer, labels = load_model_and_vectorizer()
    
    # Загружаем тестовые данные
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Предобработка тестовых текстов
    logger.info("Предобработка тестовых данных...")
    processed_texts = []
    for text in newsgroups_test.data:
        try:
            processed_tokens = preprocess(text)
            processed_text = " ".join(processed_tokens)
            processed_texts.append(processed_text)
        except Exception:
            processed_texts.append("")
    
    # Векторизация
    X_test = vectorizer.transform(processed_texts)
    y_true = newsgroups_test.target
    
    # Предсказания
    logger.info("Получение предсказаний...")
    y_pred_proba = model.predict(X_test.toarray(), verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)
    
    # Вычисление метрик
    accuracy = accuracy_score(y_true, y_pred)
    f1_macro = f1_score(y_true, y_pred, average='macro')
    f1_micro = f1_score(y_true, y_pred, average='micro')
    
    # Отчет по классам
    class_report = classification_report(
        y_true, y_pred, 
        target_names=labels,
        output_dict=True
    )
    
    # Матрица ошибок
    conf_matrix = confusion_matrix(y_true, y_pred)
    
    metrics = {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_micro': f1_micro,
        'classification_report': class_report,
        'confusion_matrix': conf_matrix,
        'labels': labels
    }
    
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"F1-macro: {f1_macro:.4f}")
    logger.info(f"F1-micro: {f1_micro:.4f}")
    
    return metrics


def generate_report() -> str:
    """
    Генерирует текстовый отчет о качестве модели.
    
    Returns:
        Форматированный отчет
    """
    metrics = evaluate_model()
    
    report = []
    report.append("=== ОТЧЕТ О КАЧЕСТВЕ МОДЕЛИ ===\n")
    
    report.append(f"Точность (Accuracy): {metrics['accuracy']:.4f}")
    report.append(f"F1-score (macro): {metrics['f1_macro']:.4f}")
    report.append(f"F1-score (micro): {metrics['f1_micro']:.4f}\n")
    
    report.append("=== МЕТРИКИ ПО КЛАССАМ ===")
    class_report = metrics['classification_report']
    
    for label in metrics['labels']:
        if label in class_report:
            precision = class_report[label]['precision']
            recall = class_report[label]['recall']
            f1 = class_report[label]['f1-score']
            support = class_report[label]['support']
            
            report.append(f"{label}:")
            report.append(f"  Precision: {precision:.3f}")
            report.append(f"  Recall: {recall:.3f}")
            report.append(f"  F1-score: {f1:.3f}")
            report.append(f"  Support: {support}")
            report.append("")
    
    # Сохраняем отчет
    report_path = Path("artifacts") / "model_evaluation_report.txt"
    report_text = "\n".join(report)
    
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    
    logger.info(f"Отчет сохранен: {report_path}")
    
    return report_text


if __name__ == "__main__":
    print(generate_report())