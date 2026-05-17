from __future__ import annotations

import logging
import time

from ber_tokenscope.settings import get_settings
from observability.logging import configure_logging


def main() -> None:
    configure_logging()
    settings = get_settings()
    logging.info(
        "BERTScope worker started",
        extra={
            "environment": settings.service.environment,
            "concurrency": settings.service.worker_concurrency,
        },
    )
    while True:
        time.sleep(30)
        logging.info("BERTScope worker heartbeat")


if __name__ == "__main__":
    main()
