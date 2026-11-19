"""version.py is the single version source; tags and the updater must agree."""
import re

from core.version_utils import parse_version


def test_version_py_is_well_formed():
    from version import __version__
    assert re.fullmatch(r'\d+(\.\d+){1,2}', __version__)


def test_branch_prefixed_release_tags_parse():
    assert parse_version("main-6.4") == (6, 4, 0)
    assert parse_version("gpu-6.4.1") == (6, 4, 1)
    assert parse_version("gpu-feat-6.10") == (6, 10, 0)


def test_prefixed_tag_newer_than_current():
    from version import __version__
    assert parse_version("main-99.0") > parse_version(__version__)
    assert not parse_version("main-0.1") > parse_version(__version__)


def test_current_version_tag_equals_current():
    from version import __version__
    assert parse_version(f"main-{__version__}") == parse_version(__version__)


def test_no_digits():
    assert parse_version("latest") == (0, 0, 0)
    assert parse_version("") == (0, 0, 0)


def test_legacy_behaviour_preserved():
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version("1.abc.3") == (1, 0, 3)
    assert parse_version("6.4-beta") == (6, 4, 0)


import pytest  # noqa: E402


@pytest.mark.parametrize("url", [
    "https://github.com/iishong0w0/Axiom-AI-Aimbot/releases/tag/main-6.4",
    "https://www.github.com/x/y",
    " https://github.com/x/y ",
])
def test_release_urls_on_github_are_safe(url):
    from core.version_utils import is_safe_release_url
    assert is_safe_release_url(url) is True


@pytest.mark.parametrize("url", [
    "http://github.com/x/y",                 # not https
    "file:///C:/Windows/System32/calc.exe",
    "javascript:alert(1)",
    "https://evil.example/github.com",
    "https://github.com.evil.example/x",
    "https://github.com@evil.example/x",     # userinfo trick: the host is evil.example
    "https://user:pw@github.com/x",
    "",
    None,
    "//github.com/x",
])
def test_other_release_urls_are_rejected(url):
    from core.version_utils import is_safe_release_url
    assert is_safe_release_url(url) is False
