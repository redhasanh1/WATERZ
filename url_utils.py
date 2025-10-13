import re
from typing import List, Tuple
from urllib.parse import urlparse, urlunparse, parse_qsl, urlencode


_WATERMARK_KEY_RE = re.compile(
    r"(^|[\W_])("
    r"watermark|water_mark|wm|logo|overlay|stamp|badge|branding|brandmark"
    r")(\W|$)",
    re.IGNORECASE,
)


def is_potentially_watermarked_url(url: str) -> bool:
    """Heuristically detect if a video URL likely includes a watermark.

    Flags URLs where any query parameter name (or value) contains watermark-like
    indicators, e.g. `watermark=1`, `wm=true`, `logo=on`, etc., or the path
    contains obvious watermark markers.
    """
    try:
        parts = urlparse(url)
        # Check path hints
        if re.search(r"watermark|water_mark|watermarked|wm/", parts.path, re.IGNORECASE):
            return True

        # Check query params
        for k, v in parse_qsl(parts.query, keep_blank_values=True):
            if _WATERMARK_KEY_RE.search(k) or _WATERMARK_KEY_RE.search(v):
                return True
            # Common simple toggles
            if k.lower() in {"wm", "watermark"} and v.lower() in {"1", "true", "on", "yes"}:
                return True
        return False
    except Exception:
        # Be conservative on parse errors
        return False


def sanitize_video_url(url: str) -> str:
    """Return a version of `url` with watermark-related params removed/disabled.

    - Removes query params that look watermark-related.
    - For simple toggles like `wm`/`watermark`, sets to `0` if removal would
      produce an empty query that could break upstream validation.
    - Does not alter path or other auth/signature params.
    """
    parts = urlparse(url)

    # Fast path if no query
    if not parts.query:
        return url

    params = parse_qsl(parts.query, keep_blank_values=True)
    kept: List[Tuple[str, str]] = []
    removed: List[Tuple[str, str]] = []

    for k, v in params:
        kl, vl = k.lower(), v.lower()
        if _WATERMARK_KEY_RE.search(k) or _WATERMARK_KEY_RE.search(v):
            removed.append((k, v))
            continue
        if kl in {"wm", "watermark"} and vl in {"1", "true", "on", "yes"}:
            removed.append((k, v))
            continue
        kept.append((k, v))

    # If everything got removed and there were watermark toggles, prefer an explicit off toggle
    if not kept and removed:
        kept = [("watermark", "0")]

    new_query = urlencode(kept, doseq=True)
    sanitized = urlunparse(parts._replace(query=new_query))
    return sanitized


def pick_best_video_url(candidates: List[str]) -> str:
    """Pick the most likely non-watermarked video URL from candidates.

    Prefers URLs that are not flagged as watermarked. If all candidates appear
    watermarked, returns a sanitized version of the first candidate.
    """
    for u in candidates:
        if not is_potentially_watermarked_url(u):
            return u
    if candidates:
        return sanitize_video_url(candidates[0])
    return ""

