"""
Directory sandboxing module.
Restricts file access to whitelisted directories only.
"""

import os
from pathlib import Path


# Base directory for the web application
WEB_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Whitelisted directories - files can ONLY be accessed within these
ALLOWED_DIRECTORIES = {
    'static': os.path.realpath(WEB_DIR),
    'css': os.path.realpath(os.path.join(WEB_DIR, 'css')),
    'js': os.path.realpath(os.path.join(WEB_DIR, 'js')),
    'uploads': os.path.realpath(os.path.join(WEB_DIR, 'temp', 'uploads')),
    'temp': os.path.realpath(os.path.join(WEB_DIR, 'temp')),
    'results': os.path.realpath(os.path.join(WEB_DIR, 'results')),
    'quarantine': os.path.realpath(os.path.join(WEB_DIR, 'quarantine')),
}


def get_allowed_directory(category: str) -> str:
    """Get the absolute path for an allowed directory category."""
    if category not in ALLOWED_DIRECTORIES:
        raise ValueError(f"Unknown directory category: {category}")
    return ALLOWED_DIRECTORIES[category]


def is_path_safe(path: str, category: str) -> bool:
    """
    Check if a path is within the allowed directory for its category.

    Args:
        path: The path to check (can be relative or absolute)
        category: The category of allowed directory ('static', 'uploads', etc.)

    Returns:
        True if path is safe, False otherwise
    """
    try:
        allowed_base = get_allowed_directory(category)
        # Resolve the path to its real absolute form (follows symlinks)
        real_path = os.path.realpath(path)
        # Check if the resolved path starts with the allowed base
        return real_path.startswith(allowed_base + os.sep) or real_path == allowed_base
    except (ValueError, OSError):
        return False


def validate_path(path: str, category: str) -> str:
    """
    Validate and return the real path if it's within allowed directory.

    Args:
        path: The path to validate
        category: The category of allowed directory

    Returns:
        The resolved real path

    Raises:
        PermissionError: If path is outside allowed directory
    """
    allowed_base = get_allowed_directory(category)
    real_path = os.path.realpath(path)

    if not (real_path.startswith(allowed_base + os.sep) or real_path == allowed_base):
        raise PermissionError(f"Access denied: path '{path}' is outside allowed directory")

    return real_path


def safe_join(category: str, *paths: str) -> str:
    """
    Safely join paths within an allowed directory.

    Args:
        category: The category of allowed directory
        *paths: Path components to join

    Returns:
        The joined and validated absolute path

    Raises:
        PermissionError: If resulting path is outside allowed directory
    """
    allowed_base = get_allowed_directory(category)
    joined = os.path.join(allowed_base, *paths)
    real_path = os.path.realpath(joined)

    if not (real_path.startswith(allowed_base + os.sep) or real_path == allowed_base):
        raise PermissionError(f"Access denied: resulting path is outside allowed directory")

    return real_path
