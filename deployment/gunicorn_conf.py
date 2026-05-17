import multiprocessing
import os

bind = f"0.0.0.0:{os.getenv('BERTSCOPE_API_PORT', '8788')}"
workers = int(os.getenv("WEB_CONCURRENCY", multiprocessing.cpu_count() or 1))
worker_class = "uvicorn.workers.UvicornWorker"
timeout = int(os.getenv("GUNICORN_TIMEOUT", "120"))
accesslog = "-"
errorlog = "-"
