"""Rules for serving the Web Control client as an installable app (PWA) — pure Python, no web framework.

``web_control_server`` applies ``static_headers()`` to every static file it serves; the tests pin the
manifest, the service worker and these headers without needing fastapi.
"""
from __future__ import annotations

from typing import Dict

MANIFEST_TYPE = "application/manifest+json"
JS_TYPE = "text/javascript"


def static_headers(path: str) -> Dict[str, str]:
    """Extra/overriding response headers for the static client file at ``path`` (relative URL path).

    * every file: ``Cache-Control: no-cache`` (revalidate; ETags keep that cheap) so the page, script
      and styles can never run out of step;
    * ``manifest.webmanifest``: the manifest media type browsers require for installability;
    * ``sw.js``: JavaScript type and ``Service-Worker-Allowed: /`` (it lives at the root, scope ``/``).
    """
    name = (path or "").lstrip("/").split("?", 1)[0].lower()
    headers = {"Cache-Control": "no-cache"}
    if name.endswith(".webmanifest"):
        headers["Content-Type"] = MANIFEST_TYPE
    elif name == "sw.js":
        headers["Content-Type"] = JS_TYPE
        headers["Service-Worker-Allowed"] = "/"
    return headers
