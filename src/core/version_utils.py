"""Pure version-string helpers (no Qt/Windows imports, so testable anywhere)."""
import re
from urllib.parse import urlparse

_LEADING_INT_RE = re.compile(r'\d+')


def parse_version(v_str):
    """Parse a version or release tag into a >=3-int tuple.

    Anything before the first digit is dropped, so 'v6.4', 'main-6.4' and
    'gpu-feat-6.4' all parse to (6, 4, 0). Each dot-separated field then
    contributes its leading digit run ('4-beta' -> 4); a field with no
    leading digits counts as 0 ('1.abc.3' -> (1, 0, 3)).
    """
    s = str(v_str).lower().strip()
    first = re.search(r'\d', s)
    s = s[first.start():] if first else ''

    parts = []
    for part in s.split('.') if s else []:
        match = _LEADING_INT_RE.match(part.strip())
        parts.append(int(match.group()) if match else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts)


_RELEASE_URL_HOSTS = frozenset({"github.com", "www.github.com"})


def is_safe_release_url(url) -> bool:
    """True only for an ``https://github.com/...`` URL.

    The update page URL comes straight out of the GitHub API response and is handed to
    ``webbrowser.open()``; anything else (``file:``, ``javascript:``, another host, a
    ``github.com@evil.example`` userinfo trick) must never reach the browser.
    """
    try:
        parts = urlparse(str(url).strip())
        return (parts.scheme == "https" and parts.hostname in _RELEASE_URL_HOSTS
                and parts.username is None and parts.password is None)
    except ValueError:
        return False
