import logging
import os
import queue
from logging.handlers import QueueHandler, QueueListener, RotatingFileHandler
from pathlib import Path

_listener: QueueListener | None = None

def configure_logging() -> None:
    global _listener

    log_dir = Path(os.environ["LOCALAPPDATA"]) / "HexaFlow" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_queue: queue.SimpleQueue[logging.LogRecord] = queue.SimpleQueue()

    file_handler = RotatingFileHandler(
        log_dir / "hexaflow.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.INFO)
    root.addHandler(QueueHandler(log_queue))

    # GenieX startup INFO callbacks should not be sent to an interactive stream.
    logging.getLogger("geniex").setLevel(logging.WARNING)
    logging.getLogger("geniex._ffi._api").setLevel(logging.WARNING)

    # Do not print a second traceback when logging itself cannot emit a record.
    logging.raiseExceptions = False

    _listener = QueueListener(
        log_queue,
        file_handler,
        respect_handler_level=True,
    )
    _listener.start()

def shutdown_logging() -> None:
    global _listener
    if _listener is not None:
        _listener.stop()
        _listener = None