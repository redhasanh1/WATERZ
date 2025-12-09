"""
MIME type validation module.
Verifies file content matches declared extension using magic bytes.
"""

import os
import struct


class MIMEValidationError(Exception):
    """Raised when MIME validation fails."""
    pass


# Magic byte signatures for allowed file types
# Format: (bytes_to_read, offset, magic_bytes, mime_type)
MAGIC_SIGNATURES = {
    # Images
    'image/png': [(8, 0, b'\x89PNG\r\n\x1a\n')],
    'image/jpeg': [(3, 0, b'\xff\xd8\xff')],
    'image/gif': [(6, 0, b'GIF87a'), (6, 0, b'GIF89a')],
    'image/webp': [(4, 8, b'WEBP')],  # After RIFF header
    'image/bmp': [(2, 0, b'BM')],
    'image/x-icon': [(4, 0, b'\x00\x00\x01\x00'), (4, 0, b'\x00\x00\x02\x00')],

    # Videos
    'video/mp4': [
        (8, 4, b'ftyp'),  # ISO Base Media file
        (8, 4, b'isom'),
        (8, 4, b'mp41'),
        (8, 4, b'mp42'),
        (8, 4, b'mmp4'),
        (8, 4, b'M4V '),
        (8, 4, b'avc1'),
    ],
    'video/quicktime': [(8, 4, b'moov'), (8, 4, b'ftyp')],
    'video/x-msvideo': [(4, 0, b'RIFF'), (4, 8, b'AVI ')],  # RIFF + AVI
    'video/x-matroska': [(4, 0, b'\x1a\x45\xdf\xa3')],  # WebM/MKV
}

# Extension to expected MIME types mapping
EXTENSION_MIME_MAP = {
    '.png': ['image/png'],
    '.jpg': ['image/jpeg'],
    '.jpeg': ['image/jpeg'],
    '.gif': ['image/gif'],
    '.webp': ['image/webp'],
    '.bmp': ['image/bmp'],
    '.ico': ['image/x-icon'],
    '.svg': ['image/svg+xml'],  # Text-based, handled separately
    '.mp4': ['video/mp4', 'video/quicktime'],
    '.mov': ['video/quicktime', 'video/mp4'],
    '.avi': ['video/x-msvideo'],
    '.mkv': ['video/x-matroska'],
    '.webm': ['video/x-matroska'],
}


def _read_bytes(filepath: str, count: int, offset: int = 0) -> bytes:
    """Read specific bytes from a file."""
    with open(filepath, 'rb') as f:
        f.seek(offset)
        return f.read(count)


def _detect_mime_type(filepath: str) -> str:
    """Detect MIME type from file content using magic bytes."""
    try:
        # Read first 32 bytes (enough for most signatures)
        header = _read_bytes(filepath, 32)

        # Check each signature type
        for mime_type, signatures in MAGIC_SIGNATURES.items():
            for sig_len, offset, magic in signatures:
                if len(header) >= offset + len(magic):
                    file_bytes = header[offset:offset + len(magic)]
                    if file_bytes == magic:
                        return mime_type

        # Special case: MP4/MOV files with ftyp box
        if len(header) >= 12:
            # ftyp box can be at various offsets
            if b'ftyp' in header:
                # Check what brand comes after ftyp
                ftyp_pos = header.find(b'ftyp')
                if ftyp_pos >= 0 and ftyp_pos + 8 <= len(header):
                    brand = header[ftyp_pos + 4:ftyp_pos + 8]
                    if brand in [b'isom', b'iso2', b'mp41', b'mp42', b'avc1', b'M4V ']:
                        return 'video/mp4'
                    if brand in [b'qt  ', b'moov']:
                        return 'video/quicktime'
                return 'video/mp4'  # Generic ftyp is usually MP4

        # Special case: AVI files (RIFF container)
        if header[:4] == b'RIFF' and len(header) >= 12:
            if header[8:12] == b'AVI ':
                return 'video/x-msvideo'

        # Check for SVG (text-based XML)
        try:
            text_header = header.decode('utf-8', errors='ignore').lower()
            if '<?xml' in text_header or '<svg' in text_header:
                return 'image/svg+xml'
        except:
            pass

        return 'application/octet-stream'

    except Exception as e:
        return 'application/octet-stream'


def validate_mime_type(filepath: str, declared_extension: str) -> bool:
    """
    Validate that file content matches its declared extension.

    Args:
        filepath: Path to the file to validate
        declared_extension: The file extension (with dot, e.g., '.jpg')

    Returns:
        True if valid

    Raises:
        MIMEValidationError: If validation fails
    """
    if not os.path.exists(filepath):
        raise MIMEValidationError("File not found")

    # Normalize extension
    ext = declared_extension.lower()
    if not ext.startswith('.'):
        ext = '.' + ext

    # Get expected MIME types for this extension
    expected_mimes = EXTENSION_MIME_MAP.get(ext)
    if not expected_mimes:
        raise MIMEValidationError(f"Unknown file extension: {ext}")

    # Detect actual MIME type
    detected_mime = _detect_mime_type(filepath)

    # SVG is text-based, be lenient
    if ext == '.svg':
        if detected_mime == 'image/svg+xml':
            return True
        # Check file content for SVG tags
        try:
            content = _read_bytes(filepath, 1024).decode('utf-8', errors='ignore').lower()
            if '<svg' in content or '<?xml' in content:
                return True
        except:
            pass
        raise MIMEValidationError(f"File content does not match SVG format")

    # Check if detected type matches expected
    if detected_mime not in expected_mimes:
        # Be lenient with video containers (MP4/MOV are often interchangeable)
        video_types = {'video/mp4', 'video/quicktime'}
        if ext in ['.mp4', '.mov'] and detected_mime in video_types:
            return True

        raise MIMEValidationError(
            f"File content mismatch: expected {expected_mimes}, detected {detected_mime}"
        )

    return True


def get_safe_extension(filepath: str) -> str:
    """
    Get the actual file extension based on content, not filename.
    Useful for sanitizing uploads.

    Args:
        filepath: Path to the file

    Returns:
        The appropriate extension based on file content
    """
    mime_type = _detect_mime_type(filepath)

    # Reverse lookup
    for ext, mimes in EXTENSION_MIME_MAP.items():
        if mime_type in mimes:
            return ext

    return ''
