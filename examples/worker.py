"""Small long-running process used to demonstrate supervisor and tmux."""

from __future__ import annotations

import logging
import signal
import threading

stop_event = threading.Event()
logger = logging.getLogger(__name__)


def request_stop(signum: int, _frame: object) -> None:
    logger.info("received signal %s, stopping", signum)
    stop_event.set()


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s worker: %(message)s",
    )
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

    iteration = 0
    logger.info("started")
    while not stop_event.wait(10):
        iteration += 1
        logger.info("heartbeat %d", iteration)
    logger.info("stopped")


if __name__ == "__main__":
    main()
