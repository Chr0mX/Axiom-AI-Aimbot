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
