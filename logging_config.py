"""
Logging Configuration for MyITS Auto-Grader

This module sets up comprehensive logging for all components of the system.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output."""

    COLORS = {
        'DEBUG': '\033[36m',  # Cyan
        'INFO': '\033[32m',  # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',  # Red
        'CRITICAL': '\033[35m',  # Magenta
        'RESET': '\033[0m'  # Reset
    }

    def format(self, record):
        if hasattr(record, 'no_color'):
            return super().format(record)

        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']

        # Add color to levelname
        record.levelname = f"{color}{record.levelname}{reset}"

        return super().format(record)


def setup_logging(log_level='INFO', log_dir='logs'):
    """
    Set up comprehensive logging for the application.

    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir: Directory to store log files
    """

    # Create logs directory
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Create timestamp for log files
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level.upper()))

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler with colors
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = ColoredFormatter(
        '%(asctime)s | %(name)-20s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # Main log file handler
    main_log_file = log_path / f'auto_grader_{timestamp}.log'
    file_handler = logging.handlers.RotatingFileHandler(
        main_log_file,
        maxBytes=10 * 1024 * 1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s | %(name)-25s | %(levelname)-8s | %(funcName)-15s | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Error log file handler (errors only)
    error_log_file = log_path / f'errors_{timestamp}.log'
    error_handler = logging.FileHandler(error_log_file)
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(file_formatter)
    root_logger.addHandler(error_handler)

    # Docker operations log
    docker_log_file = log_path / f'docker_{timestamp}.log'
    docker_handler = logging.FileHandler(docker_log_file)
    docker_handler.setLevel(logging.DEBUG)
    docker_handler.setFormatter(file_formatter)

    # Docker logger
    docker_logger = logging.getLogger('docker_operations')
    docker_logger.addHandler(docker_handler)
    docker_logger.setLevel(logging.DEBUG)

    # Plagiarism detection log
    plagiarism_log_file = log_path / f'plagiarism_{timestamp}.log'
    plagiarism_handler = logging.FileHandler(plagiarism_log_file)
    plagiarism_handler.setLevel(logging.DEBUG)
    plagiarism_handler.setFormatter(file_formatter)

    # Plagiarism logger
    plagiarism_logger = logging.getLogger('plagiarism_detection')
    plagiarism_logger.addHandler(plagiarism_handler)
    plagiarism_logger.setLevel(logging.DEBUG)

    # File processing log
    file_proc_log_file = log_path / f'file_processing_{timestamp}.log'
    file_proc_handler = logging.FileHandler(file_proc_log_file)
    file_proc_handler.setLevel(logging.DEBUG)
    file_proc_handler.setFormatter(file_formatter)

    # File processing logger
    file_proc_logger = logging.getLogger('file_processing')
    file_proc_logger.addHandler(file_proc_handler)
    file_proc_logger.setLevel(logging.DEBUG)

    # Grading log
    grading_log_file = log_path / f'grading_{timestamp}.log'
    grading_handler = logging.FileHandler(grading_log_file)
    grading_handler.setLevel(logging.DEBUG)
    grading_handler.setFormatter(file_formatter)

    # Grading logger
    grading_logger = logging.getLogger('grading')
    grading_logger.addHandler(grading_handler)
    grading_logger.setLevel(logging.DEBUG)

    # Log startup information
    main_logger = logging.getLogger('main')
    main_logger.info("=" * 60)
    main_logger.info("MyITS Auto-Grader Starting Up")
    main_logger.info(f"Log Level: {log_level}")
    main_logger.info(f"Main Log File: {main_log_file}")
    main_logger.info(f"Error Log File: {error_log_file}")
    main_logger.info(f"Docker Log File: {docker_log_file}")
    main_logger.info(f"Plagiarism Log File: {plagiarism_log_file}")
    main_logger.info(f"File Processing Log File: {file_proc_log_file}")
    main_logger.info(f"Grading Log File: {grading_log_file}")
    main_logger.info("=" * 60)

    return {
        'main_log': str(main_log_file),
        'error_log': str(error_log_file),
        'docker_log': str(docker_log_file),
        'plagiarism_log': str(plagiarism_log_file),
        'file_processing_log': str(file_proc_log_file),
        'grading_log': str(grading_log_file)
    }


def get_logger(name):
    """Get a logger with the specified name."""
    return logging.getLogger(name)


def log_function_entry(logger, func_name, **kwargs):
    """Log function entry with parameters."""
    params = ', '.join([f"{k}={v}" for k, v in kwargs.items()])
    logger.debug(f"ENTRY: {func_name}({params})")


def log_function_exit(logger, func_name, result=None, execution_time=None):
    """Log function exit with result."""
    msg = f"EXIT: {func_name}"
    if execution_time:
        msg += f" (took {execution_time:.2f}s)"
    if result is not None:
        if isinstance(result, (list, dict)):
            msg += f" -> {type(result).__name__}(len={len(result)})"
        else:
            msg += f" -> {result}"
    logger.debug(msg)


def log_error_with_traceback(logger, error, context=""):
    """Log error with full traceback."""
    import traceback
    logger.error(f"ERROR in {context}: {str(error)}")
    logger.error(f"Traceback:\n{traceback.format_exc()}")


def log_docker_operation(operation, command, result, error=None):
    """Log Docker operations."""
    docker_logger = logging.getLogger('docker_operations')
    docker_logger.info(f"DOCKER {operation}: {command}")
    if error:
        docker_logger.error(f"DOCKER ERROR: {error}")
    else:
        docker_logger.debug(f"DOCKER SUCCESS: {result[:200] if result else 'No output'}")


# Suppress noisy loggers
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('docker').setLevel(logging.WARNING)