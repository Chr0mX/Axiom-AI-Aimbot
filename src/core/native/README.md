# Vendored native binaries

## `directshow_capture.dll`

Prebuilt V1 capture DLL from
[`chr0mx/DirectShow-Capture-DLL`](https://github.com/chr0mx/DirectShow-Capture-DLL),
vendored here the same way `pygrabber`/`comtypes` wheels are vendored under
`src/python/dependencies/` — no user-side MSVC build required to use the
`directshow` capture backend (`src/core/directshow_capture.py` /
`DirectShowCapture` in `screen_capture.py`).

- Source commit: `275e1c0d2f1e5bf0f8724bc0641f1117c5b9c9a4`
  (`bin/directshow_capture.dll` at that commit)
- Supports both `DSC_PIXEL_FORMAT_NV12` and `DSC_PIXEL_FORMAT_MJPEG` — see
  that repo's `src/include/directshow_capture.h` for the full C ABI this
  wraps.

If the upstream DLL changes, rebuild it there (`cmake -B build -A x64 &&
cmake --build build --config Release`) and copy the new
`bin/directshow_capture.dll` over this file — it's a committed binary, not
generated at install time, so it goes stale silently if the source changes
without a corresponding rebuild-and-recommit. Update the commit hash above
when you do.
