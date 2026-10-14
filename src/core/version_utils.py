"""Pure version-string helpers (no Qt/Windows imports, so testable anywhere)."""
import re

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
