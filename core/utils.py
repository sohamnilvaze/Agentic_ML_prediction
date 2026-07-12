"""
Common utility functions used throughout the Agentic MIMIC Framework.

Author: Soham Vaze
"""

from __future__ import annotations

import json
import logging
import random
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import numpy as np

from core.config import (
    LOG_LEVEL,
    RANDOM_STATE,
)

# =============================================================================
# Logging Utilities
# =============================================================================

logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger = logging.getLogger("AgenticMIMIC")


def log_info(message: str) -> None:
    """Log an informational message."""
    logger.info(message)


def log_warning(message: str) -> None:
    """Log a warning message."""
    logger.warning(message)


def log_error(message: str) -> None:
    """Log an error message."""
    logger.error(message)


# =============================================================================
# Reproducibility Utilities
# =============================================================================

def set_random_seed(seed: int = RANDOM_STATE) -> None:
    """
    Set all random seeds used in the project.

    Parameters
    ----------
    seed : int
        Random seed.
    """

    random.seed(seed)
    np.random.seed(seed)

    log_info(f"Random seed set to {seed}")


# =============================================================================
# Directory Utilities
# =============================================================================

def ensure_directory_exists(directory: Path) -> None:
    """
    Create a directory if it does not already exist.

    Parameters
    ----------
    directory : Path
        Directory path.
    """

    directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# JSON Utilities
# =============================================================================

def save_json(data: Dict[str, Any], filepath: Path) -> None:
    """
    Save dictionary as JSON.

    Parameters
    ----------
    data : dict
    filepath : Path
    """

    with open(filepath, "w") as f:
        json.dump(data, f, indent=4)

    log_info(f"Saved JSON to {filepath}")


def load_json(filepath: Path) -> Dict[str, Any]:
    """
    Load JSON file.

    Parameters
    ----------
    filepath : Path

    Returns
    -------
    dict
    """

    with open(filepath, "r") as f:
        return json.load(f)


# =============================================================================
# Timestamp Utilities
# =============================================================================

def current_timestamp() -> str:
    """
    Return current timestamp.

    Returns
    -------
    str
    """

    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


# =============================================================================
# Reporting Utilities
# =============================================================================

def print_section(title: str) -> None:
    """
    Print formatted section heading.

    Parameters
    ----------
    title : str
    """

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =============================================================================
# Validation Utilities
# =============================================================================

def assert_file_exists(filepath: Path) -> None:
    """
    Raise FileNotFoundError if file does not exist.

    Parameters
    ----------
    filepath : Path
    """

    if not filepath.exists():
        raise FileNotFoundError(f"{filepath} not found.")


# =============================================================================
# Future Agent Utilities
# =============================================================================

def create_agent_response(
    status: str,
    reasoning: str,
    output: Dict[str, Any],
    logs: list[str] | None = None,
) -> Dict[str, Any]:
    """
    Standard response format for every future agent.

    Parameters
    ----------
    status : str
    reasoning : str
    output : dict
    logs : list

    Returns
    -------
    dict
    """

    if logs is None:
        logs = []

    return {
        "status": status,
        "reasoning": reasoning,
        "output": output,
        "logs": logs,
    }