"""OpenTelemetry exporter for product telemetry events."""

from __future__ import annotations

import contextlib
import json
import logging
import sys
from typing import Any

logger: logging.Logger | None = None
_PROVIDER: Any = None
_last_check_failed_at: float | None = None
"""Timestamp (monotonic) of the last failed TCP check, or None."""

_CHECK_COOLDOWN_SECONDS = 5.0
"""Minimum seconds between TCP reachability retries after a failure.

When the collector is unreachable we cache the negative result so
that rapid-fire ``emit_product_log`` calls (e.g. during service
startup) don't all hammer the network in parallel.
"""

_logger = logging.getLogger("lemoncrow.product.telemetry.otel")


class _OtelNoiseFilter(logging.Filter):
    """Silence OTel SDK / HTTP-client log messages that are harmless when the
    collector is temporarily unavailable:

    * ``Timeout was exceeded in force_flush()``
    * ``Exception while exporting logs`` + traceback
    * ``Failed to resolve`` / ``Connection refused`` etc.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        name = record.name
        return not (name.startswith("opentelemetry") or name in ("urllib3.connectionpool", "requests"))


_sdk_noise_filter = _OtelNoiseFilter()


def _apply_silence() -> None:
    """Apply the noise filter so OTel / HTTP-client messages are silenced.

    Idempotent: re-running never stacks duplicate filters. The filter is
    attached to every existing matching logger (covering loggers that carry
    their own handlers or disable propagation), to the named parent loggers
    (covering records logged directly through them), and to
    ``logging.lastResort`` (covering the lazily-created SDK child loggers
    whose records fall back to stderr when no root handler is configured) —
    so no global ``logging.getLogger`` monkeypatch is needed to catch loggers
    created later by the SDK.
    """
    for _ln, _lv in list(logging.Logger.manager.loggerDict.items()):
        if (
            isinstance(_lv, logging.Logger)
            and not any(f is _sdk_noise_filter for f in _lv.filters)
            and (_ln.startswith("opentelemetry") or _ln in ("urllib3.connectionpool", "requests"))
        ):
            _lv.addFilter(_sdk_noise_filter)

    # Attach to the named parent loggers so records logged directly through
    # them (e.g. "urllib3.connectionpool", "requests") are dropped at source.
    for nm in ("opentelemetry", "urllib3.connectionpool", "requests"):
        lg = logging.getLogger(nm)
        if not any(f is _sdk_noise_filter for f in lg.filters):
            lg.addFilter(_sdk_noise_filter)

    # Logger-level filters do NOT cascade to child loggers, and the SDK creates
    # its emitting loggers (opentelemetry.sdk._logs.*) lazily after init. With
    # no root handlers, their records fall back to ``logging.lastResort`` (the
    # implicit stderr handler). Filtering there suppresses those tracebacks for
    # all current + future descendants by name — no root handler, no getLogger
    # monkeypatch.
    if logging.lastResort is not None and not any(f is _sdk_noise_filter for f in logging.lastResort.filters):
        logging.lastResort.addFilter(_sdk_noise_filter)


_apply_silence()


def init_otel(
    *,
    endpoint: str = "http://localhost:4318",
    service_version: str = "0.1.0",
    headers: dict[str, str] | None = None,
) -> bool:
    global logger, _PROVIDER, _last_check_failed_at
    import time as _time

    if logger is not None:
        return True

    # Never build the pipeline while the interpreter is tearing down. The SDK
    # starts threads and registers its own atexit hooks during construction,
    # both of which raise RuntimeError once finalization has begun. See
    # emit_product_log for why we can still be called that late.
    if sys.is_finalizing():
        return False

    # Negative cache: if we recently failed a TCP check, skip retrying for a
    # short window to avoid hammering the network (common during service
    # startup when many requests arrive before the collector is ready).
    if _last_check_failed_at is not None and _time.monotonic() - _last_check_failed_at < _CHECK_COOLDOWN_SECONDS:
        return False

    # Quick connectivity check — avoid creating the OTel pipeline (and its
    # background BatchLogRecordProcessor thread) when the collector is not
    # reachable.  This prevents two kinds of noise at startup:
    #   • force_flush() timeout warnings during shutdown
    #   • "Exception while exporting logs" tracebacks from the worker thread
    # If the check fails, logger stays None and the next emit retries.
    endpoint_ok = _check_endpoint_reachable(endpoint)
    if not endpoint_ok:
        _last_check_failed_at = _time.monotonic()
        _logger.debug("collector not reachable at %s — telemetry deferred", endpoint)
        return False

    try:
        from opentelemetry import _logs
        from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter
        from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
        from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
        from opentelemetry.sdk.resources import Resource

        # Limit OTLP exporter retries to avoid long hangs during shutdown.
        # Default is 64s, which can lead to > 2 minutes of total backoff.
        # Setting this to 2 limits it to one attempt plus minimal backoff.
        OTLPLogExporter._MAX_RETRY_TIMEOUT = 2  # type: ignore[attr-defined]
    except Exception:
        # Telemetry is best-effort and must never write to the user's terminal:
        # this module exists to keep OTel quiet (see _apply_silence). Root-level
        # logging.exception here dumped a full traceback onto every CLI command.
        _logger.debug("otel pipeline unavailable", exc_info=True)
        return False

    from lemoncrow.core.foundation.identity import get_anon_id

    # Building the pipeline is as failure-prone as importing it — Resource.create()
    # fans out to detector threads, and the processor starts a worker — so it needs
    # the same containment. Without this, a construction failure escaped all the way
    # to the caller in emit.py and was printed there as a traceback.
    try:
        resource = Resource.create(
            {
                "service.name": "lemoncrow",
                "service.version": service_version,
                "machine.id": get_anon_id(),
            }
        )
        provider = LoggerProvider(resource=resource, shutdown_on_exit=False)
        provider.add_log_record_processor(
            BatchLogRecordProcessor(
                OTLPLogExporter(endpoint=_logs_endpoint(endpoint), timeout=2, headers=headers),
                schedule_delay_millis=1000,
                export_timeout_millis=2000,
            )
        )
        _logs.set_logger_provider(provider)
    except Exception:
        _logger.debug("otel pipeline construction failed", exc_info=True)
        return False

    logger = logging.getLogger("lemoncrow.product.telemetry.otel")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    if not any(isinstance(handler, LoggingHandler) for handler in logger.handlers):
        logger.addHandler(LoggingHandler(level=logging.DEBUG, logger_provider=provider))
    logger = logger
    _PROVIDER = provider
    _last_check_failed_at = None  # clear negative cache
    # Silence any OTel loggers that were created during the imports above.
    _apply_silence()
    return True


def _check_endpoint_reachable(endpoint: str) -> bool:
    """Return True if the OTel collector host:port is accepting TCP connections."""
    import socket

    raw = endpoint.removeprefix("http://").removeprefix("https://")
    if "/" in raw:
        raw = raw.split("/", 1)[0]
    if ":" in raw:
        host, port_str = raw.rsplit(":", 1)
    else:
        host = raw
        port_str = "4318"
    try:
        port = int(port_str)
    except ValueError:
        return False
    try:
        sock = socket.create_connection((host, port), timeout=0.5)
        sock.close()
        return True
    except OSError:
        return False


def emit_product_log(event_name: str, props: dict[str, Any]) -> bool:
    # The telemetry worker (emit.py) is a daemon thread, so it can still be
    # draining its queue after the interpreter has started finalizing. Every
    # OTel call is unsafe at that point — Resource.create() submits to a
    # ThreadPoolExecutor and raises "cannot schedule new futures after
    # interpreter shutdown". Drop the remote export; the event is still
    # recorded in the local store with exported=False.
    if sys.is_finalizing():
        return False
    if logger is None:
        from lemoncrow.core.service.telemetry.config import otel_endpoint

        if not init_otel(endpoint=otel_endpoint()):
            return False
    if logger is None:
        return False
    try:
        from opentelemetry._logs.severity import SeverityNumber

        # Flatten dict values to OTel-compatible types (str, int, float, bool)
        flat_attrs = {"event.name": event_name}
        for key, value in props.items():
            if isinstance(value, (dict, list, tuple)):
                flat_attrs[key] = json.dumps(value, ensure_ascii=False)
            else:
                flat_attrs[key] = value

        otel_logger = _PROVIDER.get_logger("lemoncrow.product.telemetry.otel")

        try:
            from opentelemetry.sdk._logs import LogRecord  # type: ignore[attr-defined]
        except ImportError:
            # opentelemetry-sdk >= 1.43 turned LogRecord into an internal ABC
            # (the public names are now ReadableLogRecord / ReadWriteLogRecord)
            # and moved the fields onto emit() itself, which reads the span
            # context from the active context — the same span get_current_span()
            # would have returned below. pyproject floats the SDK at >=1.27, so
            # both call shapes have to keep working.
            otel_logger.emit(
                body=event_name,
                attributes=flat_attrs,
                severity_text="DEBUG",
                severity_number=SeverityNumber.DEBUG,
            )
            return True

        from opentelemetry.trace import TraceFlags, get_current_span

        # Get current span context if available
        span = get_current_span()
        span_context = span.get_span_context() if span else None

        # Create LogRecord with proper span context
        record = LogRecord(
            body=event_name,
            attributes=flat_attrs,
            span_id=span_context.span_id if span_context else 1,
            trace_id=span_context.trace_id if span_context else 1,
            trace_flags=span_context.trace_flags if span_context else TraceFlags(0),
            severity_text="DEBUG",
            severity_number=SeverityNumber.DEBUG,
        )
        otel_logger.emit(record)
        return True
    except Exception:
        _logger.debug("product-log emit failed", exc_info=True)
        return False


def shutdown_otel() -> None:
    global logger, _PROVIDER, _last_check_failed_at
    provider = _PROVIDER
    logger = None
    _PROVIDER = None
    _last_check_failed_at = None  # clear negative cache
    if provider is not None:
        with contextlib.suppress(Exception):
            provider.shutdown()


def _logs_endpoint(endpoint: str) -> str:
    cleaned = endpoint.rstrip("/")
    if cleaned.endswith("/v1/logs"):
        return cleaned
    return f"{cleaned}/v1/logs"
