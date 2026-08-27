"""Utility functions for logging, file hashing, and text cleaning.

Provides helper routines used across the RAG pipeline.
"""

import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

from config import LOG_FILE, SUPPORTED_EXTENSIONS


def setup_logger(name: str = "rag_chatbot") -> logging.Logger:
    """Configure and return a file logger that records pipeline events.

    Secrets, API keys, and sensitive raw document contents are never logged.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)

        # File Handler
        file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


logger = setup_logger()


def calculate_file_hash(file_path: str | Path) -> str:
    """Calculate the SHA-256 checksum of a file for duplicate & change detection.

    Args:
        file_path: Path to the target file.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {file_path}")

    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def clean_text(text: str) -> str:
    """Clean and normalize extracted text without destroying paragraph semantics.

    - Replaces non-breaking spaces and carriage returns.
    - Consolidates redundant whitespace within lines.
    - Normalizes multiple consecutive blank lines to double newlines.
    - Strips leading and trailing whitespace.

    Args:
        text: Raw text string.

    Returns:
        Cleaned text string.
    """
    if not text:
        return ""

    # Replace carriage returns and unicode whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = text.replace("\xa0", " ").replace("\u200b", "")

    # Replace tabs and multiple horizontal spaces with a single space
    text = re.sub(r"[ \t]+", " ", text)

    # Remove trailing spaces on each line
    lines = [line.strip() for line in text.split("\n")]

    # Join lines and collapse 3+ consecutive newlines into 2
    cleaned = "\n".join(lines)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def validate_file_path(file_path: str | Path) -> tuple[bool, str]:
    """Validate whether a file exists and has a supported extension.

    Args:
        file_path: Path string or Path object.

    Returns:
        A tuple of (is_valid: bool, message: str).
    """
    try:
        path = Path(file_path).expanduser().resolve()
    except Exception as e:
        return False, f"Invalid path syntax: {e}"

    if not path.exists():
        return False, f"File does not exist: {file_path}"

    if not path.is_file():
        return False, f"Path is not a regular file: {file_path}"

    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        supported_str = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        return False, f"Unsupported file type '{path.suffix}'. Supported types: {supported_str}"

    return True, "File is valid"


def format_file_size(size_in_bytes: int) -> str:
    """Format bytes into human-readable string (KB, MB)."""
    if size_in_bytes < 1024:
        return f"{size_in_bytes} B"
    elif size_in_bytes < 1024 * 1024:
        return f"{size_in_bytes / 1024:.1f} KB"
    else:
        return f"{size_in_bytes / (1024 * 1024):.2f} MB"
