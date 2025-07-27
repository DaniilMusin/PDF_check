"""
Flask веб-приложение с двумя маршрутами:

GET  /          – форма загрузки PDF
POST /predict   – вывод результата

Для production-запуска:
    gunicorn -w 4 -b 0.0.0.0:8000 src.app.web:app
"""
from __future__ import annotations

import secrets
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from loguru import logger
from werkzeug.utils import secure_filename

from model.predict import classify

# ────────────────────────────────────────────────────────────────────────────────
ALLOWED_EXT = {".pdf"}
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

MAX_MB = 10
MAX_BYTES = MAX_MB * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
# Случайный ключ для сессий/flash-сообщений
app.secret_key = secrets.token_hex(16)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.errorhandler(413)
def too_large(_):
    flash(f"Файл превышает {MAX_MB} МБ.", "danger")
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    logger.info("Received prediction request")
    
    file = request.files.get("file")
    if file is None or file.filename == "":
        logger.warning("No file selected")
        flash("Файл не выбран.", "warning")
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        logger.warning("Invalid file type: %s", file.filename)
        flash("Разрешены только PDF файлы.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    tmp_path = TMP_DIR / filename
    logger.info("Saving file to: %s", tmp_path)
    file.save(tmp_path)

    try:
        logger.info("Starting classification for: %s", filename)
        result = classify(tmp_path, top_k=1)
        logger.info("Classification result: %s", result)
        label, prob = result[0]
        logger.info("Final result - Label: %s, Probability: %s", label, prob)
        
        # Удаляем временный файл
        tmp_path.unlink(missing_ok=True)
        
        return render_template("result.html", category=label, probability=f"{prob:.1%}")
        
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        error_msg = f"Ошибка обработки файла: {str(exc)[:100]}. Попробуйте снова."
        flash(error_msg, "danger")
        # Удаляем временный файл даже при ошибке
        tmp_path.unlink(missing_ok=True)
        return redirect(url_for("index"))


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8080, debug=True)