import logging
import logging.handlers

from collector.utils import setup_logging


def test_setup_logging(tmp_path):
    log_file = tmp_path / "test.log"
    setup_logging(level=logging.DEBUG, log_file=str(log_file))
    logger = logging.getLogger()
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)
    assert any(isinstance(h, logging.handlers.RotatingFileHandler) for h in logger.handlers)
