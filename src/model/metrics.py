"""
\u041e\u0431\u0451\u0440\u0442\u043a\u0438 \u043d\u0430\u0434 \u043c\u0435\u0442\u0440\u0438\u043a\u0430\u043c\u0438 scikit\u2011learn \u0434\u043b\u044f \u0443\u0434\u043e\u0431\u043d\u044b\u0445 \u0438\u043c\u043f\u043e\u0440\u0442\u043e\u0432
(\u0447\u0442\u043e\u0431\u044b \u043d\u0430\u0440\u0443\u0436\u043d\u044b\u0439 \u043a\u043e\u0434 \u043d\u0435 \u0437\u0430\u0432\u0438\u0441\u0435\u043b \u043d\u0430\u043f\u0440\u044f\u043c\u0443\u044e \u043e\u0442 sklearn).

\u041f\u0440\u0438\u043c\u0435\u0440:
    from model.metrics import classification_report
"""
from __future__ import annotations

from sklearn.metrics import classification_report as _cr
from sklearn.metrics import confusion_matrix as _cm

__all__ = ["classification_report", "confusion_matrix"]

classification_report = _cr
confusion_matrix = _cm
