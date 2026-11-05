"""Where Axiom keeps its files.

Everything the user creates or changes lives in one folder, ``<Axiom>/user/``:
config.json, state.json, language.json, model_notes.json, theme_colors.json,
presets/ and configs/. Paths come from this module's own location, never
from the current working directory, so starting Axiom some other way than
the launcher (a shortcut, an IDE) still finds the same files.

Shipped data and caches stay at the project root: Model/, Model_Hud/,
game.json, trt_cache/.

``AXIOM_USER_DIR`` overrides the folder (the test suite points it at a temp
directory).

Pure stdlib, no Qt — safe to import first thing from main.py.
"""
from __future__ import annotations

import logging
import os
import shutil

logger = logging.getLogger(__name__)

USER_DIR_NAME = "user"

# Files that used to sit at the project root, and their name inside user/.
_ROOT_FILES = {
    "config.json": "config.json",
    "state.json": "state.json",
    "language.json": "language.json",
    "theme_colors.json": "theme_colors.json",
    "model_info.json": "model_notes.json",
}
_ROOT_DIRS = ("presets", "configs")
_CORRUPT_PREFIX = "config.json.corrupt-"


def project_root() -> str:
    """The Axiom folder (the one holding src/, Model/ and the launcher)."""
    # this file is <root>/src/core/paths.py
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def user_dir() -> str:
    """The user-data folder, created on first use."""
    path = os.environ.get("AXIOM_USER_DIR") or os.path.join(project_root(), USER_DIR_NAME)
    os.makedirs(path, exist_ok=True)
    return path


def user_path(*parts: str) -> str:
    """A path inside the user-data folder, e.g. user_path("config.json")."""
    return os.path.join(user_dir(), *parts)


def _move_file(src: str, dst: str) -> bool:
    if os.path.exists(dst):
        logger.warning("[Paths] Kept %s at the old location: %s already exists", src, dst)
        return False
    shutil.move(src, dst)
    logger.info("[Paths] Moved %s -> %s", src, dst)
    return True


def migrate_root_user_files(root: str | None = None, dest: str | None = None) -> list[str]:
    """Move user files from the project root into user/ (one-time, idempotent).

    Never overwrites: anything already present in user/ wins and the old copy
    is left where it is (and logged). A folder (presets/, configs/) is moved
    whole when user/ doesn't have one yet, otherwise file by file. Returns the
    destination paths that were moved.
    """
    root = root or project_root()
    dest = dest or user_dir()
    os.makedirs(dest, exist_ok=True)
    if os.path.normcase(os.path.abspath(root)) == os.path.normcase(os.path.abspath(dest)):
        return []
    moved: list[str] = []

    def _try(src: str, dst: str) -> None:
        try:
            if _move_file(src, dst):
                moved.append(dst)
        except OSError as exc:
            logger.error("[Paths] Could not move %s -> %s: %s", src, dst, exc)

    for old, new in _ROOT_FILES.items():
        src = os.path.join(root, old)
        if os.path.isfile(src):
            _try(src, os.path.join(dest, new))

    try:
        names = os.listdir(root)
    except OSError:
        names = []
    for name in names:
        src = os.path.join(root, name)
        if name.startswith(_CORRUPT_PREFIX) and os.path.isfile(src):
            _try(src, os.path.join(dest, name))

    for folder in _ROOT_DIRS:
        src_dir = os.path.join(root, folder)
        if not os.path.isdir(src_dir):
            continue
        dst_dir = os.path.join(dest, folder)
        if not os.path.exists(dst_dir):
            _try(src_dir, dst_dir)
            continue
        for name in os.listdir(src_dir):
            src = os.path.join(src_dir, name)
            if os.path.isfile(src):
                _try(src, os.path.join(dst_dir, name))
        try:
            os.rmdir(src_dir)  # only succeeds once it's empty
        except OSError:
            pass
    return moved
