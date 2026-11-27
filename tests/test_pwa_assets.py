"""The Web Control client as an installable app: manifest, service worker, icons and the static-file
headers the server adds. Pure file/text checks — no browser, no web framework."""
import json
import os
import re
import struct
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from core import pwa_assets  # noqa: E402

CLIENT = os.path.join(os.path.dirname(__file__), "..", "src", "web_control_client")


def _read(name):
    with open(os.path.join(CLIENT, name), encoding="utf-8") as f:
        return f.read()


def _png_size(path):
    with open(path, "rb") as f:
        head = f.read(24)
    assert head[:8] == b"\x89PNG\r\n\x1a\n", path
    return struct.unpack(">II", head[16:24])


class TestStaticHeaders:
    def test_every_file_revalidates(self):
        for name in ("index.html", "app.js", "styles.css", "icons/icon-192.png"):
            assert pwa_assets.static_headers(name)["Cache-Control"] == "no-cache"

    def test_manifest_media_type(self):
        h = pwa_assets.static_headers("/manifest.webmanifest")
        assert h["Content-Type"] == "application/manifest+json"

    def test_service_worker_headers(self):
        h = pwa_assets.static_headers("sw.js")
        assert h["Content-Type"] == "text/javascript" and h["Service-Worker-Allowed"] == "/"
        assert "Service-Worker-Allowed" not in pwa_assets.static_headers("app.js")

    def test_query_strings_are_ignored(self):
        assert "Service-Worker-Allowed" in pwa_assets.static_headers("sw.js?v=1")


class TestManifest:
    def test_installable_fields(self):
        m = json.loads(_read("manifest.webmanifest"))
        assert m["display"] == "standalone"
        assert m["start_url"] == "./" and m["scope"] == "./"
        assert m["name"] and m["short_name"] and len(m["short_name"]) <= 12
        assert re.fullmatch(r"#[0-9a-fA-F]{6}", m["theme_color"]) and re.fullmatch(r"#[0-9a-fA-F]{6}", m["background_color"])

    def test_icons_exist_with_declared_sizes_and_a_maskable_one(self):
        m = json.loads(_read("manifest.webmanifest"))
        sizes = set()
        purposes = set()
        for ic in m["icons"]:
            w, h = _png_size(os.path.join(CLIENT, ic["src"]))
            assert f"{w}x{h}" == ic["sizes"], ic
            sizes.add(w)
            purposes.add(ic.get("purpose", "any"))
        assert {192, 512} <= sizes
        assert "maskable" in purposes and "any" in purposes

    def test_index_links_manifest_and_icons(self):
        html = _read("index.html")
        assert 'rel="manifest" href="manifest.webmanifest"' in html
        assert 'name="theme-color"' in html and "viewport-fit=cover" in html
        assert 'rel="apple-touch-icon"' in html
        assert os.path.isfile(os.path.join(CLIENT, "icons", "apple-touch-icon.png"))
        assert _png_size(os.path.join(CLIENT, "icons", "apple-touch-icon.png")) == (180, 180)


class TestServiceWorker:
    def test_asset_version_is_in_step_with_the_page(self):
        sw = _read("sw.js")
        ver = re.search(r'SHELL_VERSION = "(\d+)"', sw).group(1)
        assert f"styles.css?v={ver}" in _read("index.html")
        assert f"app.js?v={ver}" in _read("index.html")
        assert f"--axiom-css: {ver};" in _read("styles.css")
        assert f'cssVer !== "{ver}"' in _read("app.js")

    def test_every_precached_file_exists(self):
        sw = _read("sw.js")
        shell = re.search(r"var SHELL = \[(.*?)\];", sw, re.S).group(1)
        for url in re.findall(r'"([^"]+)"', shell):
            if url.startswith("./") and len(url) == 2:
                continue
            rel = url.split("?")[0]
            if "SHELL_VERSION" in url:
                continue
            assert os.path.isfile(os.path.join(CLIENT, rel)), url
        for rel in ("styles.css", "app.js"):
            assert os.path.isfile(os.path.join(CLIENT, rel))

    def test_api_requests_are_never_handled(self):
        sw = _read("sw.js")
        assert re.search(r"isApi\(url\)\)\s*return", sw), "api requests must fall through to the network"
        assert 'indexOf("/api/") === 0' in sw
        assert "req.method !== \"GET\"" in sw

    def test_old_caches_are_dropped_and_update_is_user_driven(self):
        sw = _read("sw.js")
        assert "caches.delete" in sw and "SKIP_WAITING" in sw
        # install must not skipWaiting on its own: the page offers the reload
        install = sw.split('addEventListener("install"')[1].split('addEventListener("activate"')[0]
        assert "self.skipWaiting()" not in install

    def test_offline_page_exists_and_retries(self):
        html = _read("offline.html")
        assert "Retry" in html and "setInterval" in html


class TestClientWiring:
    def test_registration_is_secure_context_only(self):
        js = _read("app.js")
        reg = js.split("function registerServiceWorker()")[1].split("\n  }\n")[0]
        assert '"serviceWorker" in navigator' in reg and "window.isSecureContext" in reg
        assert reg.index("isSecureContext") < reg.index("register(")
        assert 'register("sw.js"' in js

    def test_appearance_controls_exist(self):
        html = _read("index.html")
        for needle in ('id="theme-toggle"', 'name="ui-theme"', 'id="accent-swatches"', 'id="accent-hex"',
                       'id="motion-toggle"', 'id="other-install-btn"', 'class="tab-capsule"'):
            assert needle in html, needle
        assert html.count('class="tab-ico"') == 10

    def test_every_tab_has_an_icon_symbol(self):
        html = _read("index.html")
        tabs = re.findall(r'data-tab="(\w+)" role="tab"', html)
        assert len(tabs) == 10
        for t in tabs:
            assert f'id="i-{t}"' in html, t


class TestAppShell:
    """iOS Safari paints the scrolling *document* behind its translucent status bar, so the document is
    locked and one inner element (#scroll-root) scrolls (styles.css section 9)."""

    def test_scroll_root_wraps_sidebar_and_main_but_not_the_topbar(self):
        html = _read("index.html")
        assert html.count('id="scroll-root"') == 1
        app = html.split('<div class="app">')[1]
        root_at = app.index('id="scroll-root"')
        assert app.index('class="topbar"') < root_at                       # topbar stays above the scroller
        inner = app[root_at:app.index('/#scroll-root')]
        assert 'id="sidebar"' in inner and 'class="main"' in inner
        assert 'class="topbar"' not in inner
        assert app.index('id="toast-region"') > app.index('/#scroll-root')   # fixed overlays stay outside it

    def test_document_is_locked_and_the_inner_element_scrolls(self):
        css = _read("styles.css")
        shell = css.split("9. App shell")[1]
        html_body = re.search(r"html, body \{([^}]*)\}", shell).group(1)
        assert "100dvh" in html_body and "overflow: hidden" in html_body and "overscroll-behavior: none" in html_body
        scroller = re.search(r"#scroll-root \{([^}]*)\}", shell).group(1)
        assert "overflow-y: auto" in scroller and "-webkit-overflow-scrolling: touch" in scroller
        assert "position: relative" in re.search(r"\.topbar \{([^}]*)\}", shell).group(1)   # no longer sticky over content
        assert "overflow: visible" in re.search(r"\.main \{([^}]*)\}", shell).group(1)       # not a second scroller

    def test_studio_body_keeps_a_solid_background_colour_after_its_gradient(self):
        css = _read("styles.css")
        shell = css.split("9. App shell")[1]
        assert re.search(r':root\[data-ui="studio"\] body,\s*:root\[data-ui="studio"\]\[data-theme="light"\] body \{[^}]*background-color: var\(--bg\)', shell)
        assert "viewport-fit=cover" in _read("index.html")

    def test_tab_switch_resets_the_inner_scroller_and_theme_meta_follows_the_background(self):
        js = _read("app.js")
        assert 'getElementById("scroll-root")' in js.split("function activateTab(name)")[1].split("setLocalJSON(LAST_TAB_KEY")[0]
        meta = js.split("function updateThemeMeta()")[1].split("var btn")[0]
        assert '"--bg"' in meta
