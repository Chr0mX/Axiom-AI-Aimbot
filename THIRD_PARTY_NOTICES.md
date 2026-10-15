# Third-Party Notices

Axiom AI ships a vendored Windows Python runtime and some prebuilt binaries.
This file is an inventory to help with licence review. It is **not legal
advice**; confirm each licence against the upstream project before
redistributing. The exact vendored Python packages are listed in
`requirements-vendored.txt`.

## Bundled binaries

| Item | Location | Notes |
|---|---|---|
| FFmpeg | `src/ffmpeg/ffmpeg.exe` | The binary's embedded build string contains `--enable-gpl` and `--enable-version3`, i.e. a GPL v3 build. Provenance/source offer not recorded in the repo. |
| ddxoft driver DLL | `src/ddxoft.dll` | Provenance and licence not recorded. |
| DirectShow-Capture-DLL | `src/python/dependencies/directshow_capture.dll` | Built from `chr0mx/DirectShow-Capture-DLL`. |
| ViGEmBus installer | repo root / installers | Upstream: nefarius/ViGEmBus. |
| Microsoft VC++ Redistributable | installers | Microsoft redistribution terms. |

## Notable Python dependencies

| Package | Licence (upstream) |
|---|---|
| PyQt6 | GPL v3 / Riverbank commercial |
| PyQt6-Fluent-Widgets | GPL v3 (commercial licence available) |
| onnxruntime / onnxruntime-directml | MIT |
| numpy, opencv, pillow, mss, psutil, sympy | BSD / MIT / HPND-style |
| fastapi, starlette, uvicorn, pydantic | MIT / BSD |

## Open question for the maintainer

The project's own licence (`LICENSE.txt`, PolyForm Noncommercial) is not
GPL-compatible, while PyQt6 and PyQt6-Fluent-Widgets are GPL v3 (unless a
commercial licence is held) and the bundled FFmpeg is a GPL v3 build.
Distributing them together may be a licence conflict. Resolve by obtaining
commercial licences, replacing the GPL components, or relicensing, and record
the decision here.
