"""Audit trail (C-199): who asked what, what was retrieved, what was answered.

Written as structured events on the `audit` logger, so they land in App Insights
(traces | where message == "audit") and can be exported to a long-retention store
(Log Analytics data export / archive tier) per your retention policy (C-011).
Question text is included only when LOG_QUESTION_TEXT=true (PII, C-196).
"""

from typing import Any

from app.core.logging import get_logger

_audit = get_logger("audit")


def audit(action: str, *, user_id: str, outcome: str = "ok", **details: Any) -> None:
    _audit.info("audit", action=action, user_id=user_id, outcome=outcome, **details)
