"""
Path traversal protection module.
Validates all user-provided paths to prevent directory traversal attacks.
"""

import os
import re
from urllib.parse import unquote
from .sandbox import safe_join, WEB_DIR


class SecurityError(Exception):
    """Raised when a security violation is detected."""
    pass


# Patterns that indicate path traversal attempts
DANGEROUS_PATTERNS = [
    r'\.\./',           # Unix-style parent directory
    r'\.\.\/',          # Escaped Unix-style
    r'\.\.\\',          # Windows-style parent directory
    r'%2e%2e',          # URL encoded ..
    r'%252e%252e',      # Double URL encoded ..
    r'%c0%ae',          # Overlong UTF-8 encoding of .
    r'%c1%9c',          # Overlong UTF-8 encoding of /
    r'\x00',            # Null byte injection
    r'%00',             # URL encoded null byte
]

# Compile patterns for efficiency
DANGEROUS_REGEX = re.compile('|'.join(DANGEROUS_PATTERNS), re.IGNORECASE)


def has_dangerous_patterns(path: str) -> bool:
    """Check if path contains any dangerous patterns."""
    # URL decode the path to catch encoded attacks
    decoded = unquote(unquote(path))  # Double decode for double-encoding

    # Check original and decoded paths
    return bool(DANGEROUS_REGEX.search(path) or DANGEROUS_REGEX.search(decoded))


def validate_static_path(user_path: str, subdir: str, allowed_extensions: list) -> str:
    """
    Validate a user-provided path for static file serving.

    Args:
        user_path: The path provided by the user (from URL)
        subdir: The subdirectory to serve from ('css', 'js', or '' for root)
        allowed_extensions: List of allowed file extensions (e.g., ['.css', '.js'])

    Returns:
        The safe absolute path to the file

    Raises:
        SecurityError: If path validation fails
    """
    # Step 1: Check for dangerous patterns
    if has_dangerous_patterns(user_path):
        raise SecurityError("Invalid path: dangerous pattern detected")

    # Step 2: Check for absolute paths (shouldn't start with / or drive letter)
    if user_path.startswith('/') or user_path.startswith('\\'):
        raise SecurityError("Invalid path: absolute paths not allowed")
    if len(user_path) > 1 and user_path[1] == ':':
        raise SecurityError("Invalid path: drive letters not allowed")

    # Step 3: Normalize and check for .. after normalization
    normalized = os.path.normpath(user_path)
    if '..' in normalized:
        raise SecurityError("Invalid path: parent directory reference detected")

    # Step 4: Check file extension
    ext = os.path.splitext(user_path)[1].lower()
    if ext not in allowed_extensions:
        raise SecurityError(f"Invalid file type: {ext}")

    # Step 5: Use sandbox to construct safe path
    try:
        if subdir:
            safe_path = safe_join(subdir, normalized)
        else:
            safe_path = safe_join('static', normalized)
    except PermissionError:
        raise SecurityError("Access denied: path outside allowed directory")

    # Step 6: Verify file exists
    if not os.path.isfile(safe_path):
        raise SecurityError("File not found")

    return safe_path


def validate_upload_filename(filename: str, allowed_extensions: set) -> str:
    """
    Validate an uploaded filename.

    Args:
        filename: The original filename from the upload
        allowed_extensions: Set of allowed extensions (without dots)

    Returns:
        The validated filename (still needs secure_filename() applied)

    Raises:
        SecurityError: If filename validation fails
    """
    if not filename:
        raise SecurityError("Empty filename")

    # Check for dangerous patterns
    if has_dangerous_patterns(filename):
        raise SecurityError("Invalid filename: dangerous pattern detected")

    # Check extension
    if '.' not in filename:
        raise SecurityError("Invalid filename: no extension")

    ext = filename.rsplit('.', 1)[1].lower()
    if ext not in allowed_extensions:
        raise SecurityError(f"Invalid file type: {ext}")

    return filename
