"""UVC capture "engine": one choice that replaces two switches.

The DirectShow path of the UVC backend can run on three engines:

* ``opencv``  — ``cv2.VideoCapture(CAP_DSHOW)`` (V1, the default)
* ``native``  — the native ``directshow_capture.dll`` (V2)
* ``ffmpeg``  — an external ffmpeg.exe subprocess (V1 + FFmpeg)

They used to be exposed as two independent switches (``uvc_dshow_backend`` and
``uvc_ffmpeg_enabled``) whose illegal combination (V2 + FFmpeg) the GUI had to
patch after the fact. The persisted fields are unchanged — this module only maps
them to and from the single choice, so existing config files load as before, and
the runtime keeps reading the same two fields.

Pure Python (no Qt, no OpenCV): shared by the Qt Capture page and Web Control.
"""
from __future__ import annotations

ENGINES = ('opencv', 'native', 'ffmpeg')

# Formats the native DLL implements (dshow_capture_native / screen_capture);
# the other engines accept the full list.
NATIVE_FORMATS = ('mjpeg', 'nv12')
ALL_FORMATS = ('mjpeg', 'yuy2', 'nv12', 'yuv420p')


def is_dshow(config) -> bool:
    return str(getattr(config, 'uvc_capture_method', 'msmf')).strip().lower() == 'dshow'


def saved_engine(config) -> str:
    """The engine the two saved fields describe, whatever the capture method is (so the
    choice survives switching to MSMF and back). V2 wins over FFmpeg: the runtime gates
    FFmpeg to dshow + v1."""
    if str(getattr(config, 'uvc_dshow_backend', 'v1')).strip().lower() == 'v2':
        return 'native'
    if bool(getattr(config, 'uvc_ffmpeg_enabled', False)):
        return 'ffmpeg'
    return 'opencv'


def engine_from_config(config) -> str:
    """The engine in effect: only DirectShow has a choice; MSMF/any are always OpenCV."""
    return saved_engine(config) if is_dshow(config) else 'opencv'


def apply_engine(config, engine: str) -> str:
    """Write the two persisted fields for ``engine`` (an unknown value means ``opencv``).
    Returns the engine that was applied."""
    engine = engine if engine in ENGINES else 'opencv'
    config.uvc_dshow_backend = 'v2' if engine == 'native' else 'v1'
    config.uvc_ffmpeg_enabled = engine == 'ffmpeg'
    return engine


def video_format_choices(engine: str) -> tuple:
    """Formats offered for ``engine`` (the native DLL only implements MJPEG and NV12)."""
    return NATIVE_FORMATS if engine == 'native' else ALL_FORMATS


def effective_video_format(config) -> str:
    """The format actually usable now: the saved choice, or MJPEG when the native engine
    cannot do it. The saved value is never rewritten, so leaving the native engine brings
    the user's original choice back."""
    fmt = str(getattr(config, 'uvc_video_format', 'mjpeg') or 'mjpeg').strip().lower()
    return fmt if fmt in video_format_choices(engine_from_config(config)) else 'mjpeg'
