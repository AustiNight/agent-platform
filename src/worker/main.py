"""
Task worker process.

Runs as a separate process to handle background task execution.
For MVP, we use FastAPI's BackgroundTasks.
This module provides an alternative Redis-based worker for production.
"""

import asyncio
import signal
import sys
from typing import Any

import structlog
from redis import Redis
from rq import Worker, Queue, Connection

from src.core.config import settings

logger = structlog.get_logger()

# Flag for graceful shutdown
shutdown_requested = False


def handle_shutdown(signum: int, frame: Any) -> None:
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info("shutdown_requested", signal=signum)
    shutdown_requested = True


def main() -> None:
    """
    Main worker entry point.
    
    Connects to Redis and processes tasks from the queue.
    """
    # Set up signal handlers
    signal.signal(signal.SIGTERM, handle_shutdown)
    signal.signal(signal.SIGINT, handle_shutdown)
    
    logger.info("starting_worker", redis_url=settings.redis_url)
    
    # Connect to Redis
    redis_conn = Redis.from_url(settings.redis_url)
    
    # Create queue
    queue = Queue("tasks", connection=redis_conn)
    
    # Start worker
    with Connection(redis_conn):
        worker = Worker(
            [queue],
            name="agent-worker",
        )
        
        logger.info("worker_started", queue="tasks")
        
        # Run until shutdown
        worker.work(
            with_scheduler=False,
            burst=False,
        )


if __name__ == "__main__":
    main()
