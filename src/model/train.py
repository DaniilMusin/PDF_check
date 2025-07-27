"""
Обучение нейронной сети для классификации текстов PDF документов.

Использует dataset 20 Newsgroups и создает модель на Keras/TensorFlow.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from preprocessing.cleaner import preprocess

__all__ = ["train_model", "create_model", "prepare_data"]

# Конфигурация модели
MODEL_CONFIG = {
    "max_features": 5000,
    "epochs": 10,
    "batch_size": 32,
    "validation_split": 0.2,
    "learning_rate": 0.001,
    "dropout_rate": 0.3
}

ARTIFACTS_DIR = Path("artifacts")
ARTIFACTS_DIR.mkdir(exist_ok=True)


def prepare_data() -> Tuple[np.ndarray, np.ndarray, list[str], object]:
    """
    Загружает и подготавливает данные 20 Newsgroups для обучения.
    
    Returns:
        X: Векторизованные тексты
        y: Метки классов
        labels: Названия категорий
        vectorizer: Обученный векторизатор
    """
    logger.info("Загрузка датасета 20 Newsgroups...")
    
    # Загружаем данные
    newsgroups_train = fetch_20newsgroups(
        subset='train',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    newsgroups_test = fetch_20newsgroups(
        subset='test',
        remove=('headers', 'footers', 'quotes'),
        random_state=42
    )
    
    # Объединяем train и test для большего объема данных
    texts = list(newsgroups_train.data) + list(newsgroups_test.data)
    targets = list(newsgroups_train.target) + list(newsgroups_test.target)
    
    logger.info(f"Загружено {len(texts)} документов в {len(newsgroups_train.target_names)} категориях")
    
    # Предобработка текстов
    logger.info("Предобработка текстов...")
    processed_texts = []
    for i, text in enumerate(texts):
        if i % 1000 == 0:
            logger.debug(f"Обработано {i}/{len(texts)} документов")
        
        # Используем нашу функцию предобработки
        try:
            processed_tokens = preprocess(text)
            processed_text = " ".join(processed_tokens)
            processed_texts.append(processed_text)
        except Exception as e:
            logger.warning(f"Ошибка обработки документа {i}: {e}")
            processed_texts.append("")  # Пустой текст для проблемных документов
    
    # Векторизация с TF-IDF
    logger.info("Векторизация текстов с TF-IDF...")
    vectorizer = TfidfVectorizer(
        max_features=MODEL_CONFIG["max_features"],
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.7,
        stop_words='english'
    )
    
    X = vectorizer.fit_transform(processed_texts)
    y = np.array(targets)
    
    logger.info(f"Форма данных: X={X.shape}, y={y.shape}")
    logger.info(f"Размер словаря: {len(vectorizer.vocabulary_)}")
    
    return X.toarray(), y, newsgroups_train.target_names, vectorizer


def create_model(input_dim: int, num_classes: int) -> tf.keras.Model:
    """
    Создает нейронную сеть для классификации текста.
    
    Args:
        input_dim: Размерность входных данных
        num_classes: Количество классов
        
    Returns:
        Скомпилированная модель Keras
    """
    logger.info(f"Создание модели: input_dim={input_dim}, num_classes={num_classes}")
    
    model = Sequential([
        Dense(512, activation='relu', input_shape=(input_dim,)),
        Dropout(MODEL_CONFIG["dropout_rate"]),
        Dense(256, activation='relu'),
        Dropout(MODEL_CONFIG["dropout_rate"]),
        Dense(128, activation='relu'),
        Dropout(MODEL_CONFIG["dropout_rate"]),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=MODEL_CONFIG["learning_rate"]),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    logger.info("Архитектура модели:")
    model.summary(print_fn=logger.info)
    
    return model


def train_model() -> None:
    """
    Полный цикл обучения модели и сохранения артефактов.
    """
    logger.info("Начало обучения модели...")
    
    # Подготовка данных
    X, y, labels, vectorizer = prepare_data()
    
    # Разделение на train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Создание модели
    model = create_model(X_train.shape[1], len(labels))
    
    # Callbacks для обучения
    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-6
        )
    ]
    
    # Обучение
    logger.info("Начало обучения...")
    history = model.fit(
        X_train, y_train,
        epochs=MODEL_CONFIG["epochs"],
        batch_size=MODEL_CONFIG["batch_size"],
        validation_split=MODEL_CONFIG["validation_split"],
        callbacks=callbacks,
        verbose=1
    )
    
    # Оценка на тестовой выборке
    logger.info("Оценка модели на тестовых данных...")
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
    logger.info(f"Тестовая точность: {test_accuracy:.4f}")
    logger.info(f"Тестовая потеря: {test_loss:.4f}")
    
    # Сохранение артефактов
    logger.info("Сохранение модели и артефактов...")
    
    # Сохранение модели
    model_path = ARTIFACTS_DIR / "model.keras"
    model.save(model_path)
    logger.info(f"Модель сохранена: {model_path}")
    
    # Сохранение векторизатора
    vectorizer_path = ARTIFACTS_DIR / "vectorizer.pkl"
    with open(vectorizer_path, "wb") as f:
        pickle.dump(vectorizer, f)
    logger.info(f"Векторизатор сохранен: {vectorizer_path}")
    
    # Сохранение меток классов
    labels_path = ARTIFACTS_DIR / "labels.pkl"
    with open(labels_path, "wb") as f:
        pickle.dump(labels, f)
    logger.info(f"Метки классов сохранены: {labels_path}")
    
    # Сохранение истории обучения
    history_path = ARTIFACTS_DIR / "training_history.pkl"
    with open(history_path, "wb") as f:
        pickle.dump(history.history, f)
    logger.info(f"История обучения сохранена: {history_path}")
    
    logger.success("Обучение завершено успешно!")
    
    return model, vectorizer, labels


if __name__ == "__main__":
    train_model()