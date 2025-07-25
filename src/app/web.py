"""
Flask\u2011\u0432\u0435\u0431\u2011\u043f\u0440\u0438\u043b\u043e\u0436\u0435\u043d\u0438\u0435 \u0441 \u0434\u0432\u0443\u043c\u044f \u043c\u0430\u0440\u0448\u0440\u0443\u0442\u0430\u043c\u0438:

GET  /          \u2013 \u0444\u043e\u0440\u043c\u0430 \u0437\u0430\u0433\u0440\u0443\u0437\u043a\u0438 PDF
POST /predict   \u2013 \u0432\u044b\u0432\u043e\u0434 \u0440\u0435\u0437\u0443\u043b\u044c\u0442\u0430\u0442\u0430

\u0414\u043b\u044f production\u2011\u0437\u0430\u043f\u0443\u0441\u043a\u0430:
    gunicorn -w 4 -b 0.0.0.0:8000 src.app.web:app
"""
from __future__ import annotations

import secrets
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for
from loguru import logger
from werkzeug.utils import secure_filename

from model.predict import classify

# \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
ALLOWED_EXT = {".pdf"}
TMP_DIR = Path("tmp")
TMP_DIR.mkdir(exist_ok=True)

MAX_MB = 10
MAX_BYTES = MAX_MB * 1024 * 1024

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES
# \u0421\u043b\u0443\u0447\u0430\u0439\u043d\u044b\u0439 \u043a\u043b\u044e\u0447 \u0434\u043b\u044f \u0441\u0435\u0441\u0441\u0438\u0439/flash\u2011\u0441\u043e\u043e\u0431\u0449\u0435\u043d\u0438\u0439
app.secret_key = secrets.token_hex(16)


def _allowed(filename: str) -> bool:
    return Path(filename).suffix.lower() in ALLOWED_EXT


@app.errorhandler(413)
def too_large(_):
    flash(f"\u0424\u0430\u0439\u043b \u043f\u0440\u0435\u0432\u044b\u0448\u0430\u0435\u0442 {MAX_MB}\u00a0\u041c\u0411.", "danger")
    return redirect(url_for("index"))


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("file")
    if file is None or file.filename == "":
        flash("\u0424\u0430\u0439\u043b \u043d\u0435 \u0432\u044b\u0431\u0440\u0430\u043d.", "warning")
        return redirect(url_for("index"))

    if not _allowed(file.filename):
        flash("\u0420\u0430\u0437\u0440\u0435\u0448\u0435\u043d\u044b \u0442\u043e\u043b\u044c\u043a\u043e PDF.", "danger")
        return redirect(url_for("index"))

    filename = secure_filename(file.filename)
    tmp_path = TMP_DIR / filename
    file.save(tmp_path)

    try:
        label, prob = classify(tmp_path, top_k=1)[0]
    except Exception as exc:  # noqa: BLE001
        logger.exception("Prediction failed: %s", exc)
        flash("\u041e\u0448\u0438\u0431\u043a\u0430 \u043f\u0440\u0438 \u043e\u0431\u0440\u0430\u0431\u043e\u0442\u043a\u0435 \u0444\u0430\u0439\u043b\u0430.", "danger")
        return redirect(url_for("index"))
    finally:
        tmp_path.unlink(missing_ok=True)

    return render_template("result.html", category=label, probability=f"{prob:.1%}")
