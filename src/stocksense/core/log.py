from __future__ import annotations

import json
import logging
import re
import sys
from collections.abc import Iterable
from typing import Any

from stocksense.core.clock import Clock
from stocksense.core.config import Settings
from stocksense.core.paths import Paths

_SECRET_KEY = re.compile(r"(?i)(token|secret|pin|password|totp|authorization|api_key|jwt)")
_MASK = "***"
_RESERVED = {"ts", "level", "logger", "event"}


class StructLogger:
    """Thin wrapper: `log.info("event", key=value, ...)` becomes one JSON line."""

    def __init__(self, logger: logging.Logger) -> None:
        self._logger = logger

    def _log(self, level: int, event: str, fields: dict[str, Any], exc_info: bool = False) -> None:
        if self._logger.isEnabledFor(level):
            self._logger.log(level, event, extra={"fields": fields}, exc_info=exc_info, stacklevel=3)

    def debug(self, event: str, **fields: Any) -> None:
        self._log(logging.DEBUG, event, fields)

    def info(self, event: str, **fields: Any) -> None:
        self._log(logging.INFO, event, fields)

    def warning(self, event: str, **fields: Any) -> None:
        self._log(logging.WARNING, event, fields)

    def error(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields)

    def exception(self, event: str, **fields: Any) -> None:
        self._log(logging.ERROR, event, fields, exc_info=True)


def get_logger(name: str) -> StructLogger:
    """Structured logger; fields are redacted by key name and known secret values."""
    return StructLogger(logging.getLogger(name))


class RedactFilter(logging.Filter):
    """Masks secret-looking field keys and any known secret value inside message/field strings."""

    def __init__(self, secret_values: Iterable[str] = ()) -> None:
        super().__init__()
        # longest first so a secret containing another is replaced whole
        self._values = sorted({v for v in secret_values if v}, key=len, reverse=True)

    def _scrub(self, text: str) -> str:
        for v in self._values:
            if v in text:
                text = text.replace(v, _MASK)
        return text

    def _clean(self, value: Any) -> Any:
        if isinstance(value, str):
            return self._scrub(value)
        if isinstance(value, dict):
            return {k: _MASK if _SECRET_KEY.search(str(k)) else self._clean(v) for k, v in value.items()}
        if isinstance(value, list | tuple):
            return type(value)(self._clean(v) for v in value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = self._scrub(record.getMessage())
        record.args = None
        fields = getattr(record, "fields", None)
        if fields:
            record.fields = self._clean(fields)
        if record.exc_info and not record.exc_text:
            record.exc_text = logging.Formatter().formatException(record.exc_info)
        if record.exc_text:
            record.exc_text = self._scrub(record.exc_text)
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line: ts, level, logger, event, then the fields."""

    def format(self, record: logging.LogRecord) -> str:
        ts = self.formatTime(record, "%Y-%m-%dT%H:%M:%S")
        out: dict[str, Any] = {
            "ts": f"{ts}.{int(record.msecs):03d}",
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        for k, v in (getattr(record, "fields", None) or {}).items():
            out[k if k not in _RESERVED else f"field_{k}"] = v
        if record.exc_info and not record.exc_text:
            record.exc_text = self.formatException(record.exc_info)
        if record.exc_text:
            out["exc"] = record.exc_text
        return json.dumps(out, ensure_ascii=False, default=str)


def configure_logging(
    settings: Settings, clock: Clock, console: bool = True, level: int = logging.INFO
) -> logging.Handler:
    """Install JSON handlers on the root logger: `logs/daemon-<clock date>.log` (+ stderr). Idempotent.

    The file name carries the date at configure time; the daemon calls this again at midnight.
    Returns the file handler.
    """
    paths = Paths(settings)
    paths.logs.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for h in list(root.handlers):
        if getattr(h, "_stocksense", False):
            root.removeHandler(h)
            h.close()
    redact = RedactFilter(settings.secrets.secret_values())
    fmt = JsonFormatter()
    file_handler = logging.FileHandler(paths.logs / f"daemon-{clock.today():%Y-%m-%d}.log", encoding="utf-8")
    handlers: list[logging.Handler] = [file_handler]
    if console:
        handlers.append(logging.StreamHandler(sys.stderr))
    for h in handlers:
        h.setFormatter(fmt)
        h.addFilter(redact)
        h._stocksense = True  # type: ignore[attr-defined]
        root.addHandler(h)
    root.setLevel(level)
    return file_handler
