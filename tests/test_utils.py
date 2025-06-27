import logging
import logging.handlers

from collector.utils import setup_logging


def test_setup_logging(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logging(
        level=logging.DEBUG,
        log_file=str(log_file),
        max_bytes=1024,
        backup_count=2,
        console_output=False,
    )
    logger = logging.getLogger()
    file_handlers = [
        h for h in logger.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
    ]
    assert len(file_handlers) == 1
    handler = file_handlers[0]
    assert handler.maxBytes == 1024
    assert handler.backupCount == 2
    logger.info("hello")
    with open(log_file, "r") as f:
        data = f.read()
    assert "hello" in data
