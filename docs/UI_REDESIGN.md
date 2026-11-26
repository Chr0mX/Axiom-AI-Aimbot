# UI redesign — glass interface, floating navigation, installable web client

Branch `ui-redesign`. The brief: bring the Qt app to the polish, layout and interaction fluidity of a
reference Qt app seen in a screen recording ("the reference"), without removing or rewiring any setting,
and give the Web Control client the same look as an installable PWA.

Screenshots (ours only) are in `docs/ui_redesign/`: `compare_*` are before | after pairs, `after_*` are
single states, `web_*` are the Web Control client.

## 1. What the reference does

Read from 29 sampled frames and the tab-change / theme-change sequences.

| Area | Behaviour |
|---|---|
| Shell | Frameless window. Logo and wordmark on the left, a floating rounded **pill navigation** centred in the title row, round icon buttons on the right (theme, minimise, close). Dark navy backdrop with a violet glow. |
| Navigation | Icon + label per item. The active item is a filled accent-gradient capsule with a soft glow. The pill can be docked at the top (horizontal) or left (vertical icon rail). A small status pill (CPU, preset buttons, loaded preset) sits at the bottom edge. |
| Pages | Group title with a small accent bar, then one large (≈20 px radius) translucent card. Rows have a bold title, a muted two-line description, the control on the right, hairline dividers and a hover wash. Two equal columns on wide windows. |
| Controls | Pill value badge (`0.88`, `183px`) beside a gradient-filled slider with a round thumb; large pill toggle; flat drop-down; small ghost buttons. |
| Motion | Tab change fades the page while the capsule slides to the new item (~200 ms). Theme change is a smooth cross-fade between dark navy and pale lavender glass. Accent change recolours everything live. Settings offer accent colour and a UI-animations switch. |

## 2. Audit of the existing UI

- `AxiomWindow(FluentWindow)` with a left `NavigationInterface` (10 pages, 4 bottom action items), a 48 px
  title bar, qfluentwidgets' 300 ms pop-up page animation and a 1 s config-sync timer.
- `_applyThemeStyles()` made containers transparent and monkey-patched `paintEvent` of every
  `SettingCard`. No first-party animation code existed; the accent was a fixed blue.
- ~48 setting groups and ~226 cards on 10 pages were built from vendored `SettingCard`,
  `SwitchSettingCard` and the three slider cards. Nothing subclassed `SettingCard`, so one drop-in subclass
  covers every page.
- qfluentwidgets' stylesheets are compiled resources, so only per-widget `setCustomStyleSheet` works;
  `SettingCard`, `Slider` and the switch indicator are custom-painted and have to be subclassed.
- Pages are one single-column stack, 36 px margins, 32 px title.

## 3. Architecture

`FluentWindow` stays (frameless, native snap-layout hit-testing, `addSubInterface`/`switchTo`/qrouter, the
remote-conversion hook). Only the chrome is replaced: its `NavigationInterface` is hidden and our widgets
are laid over the title row. No page is recreated; every `setConfig`, `refreshFromConfig` and
`retranslateUi` path is unchanged.

Pure, Qt-free rules live in `src/core/` so they are unit-tested on any OS: `ui_accent.py`, `ui_motion.py`,
`ui_layout.py`, `ui_status.py`, `pwa_assets.py`.

### Components (`src/gui/fluent_app/components/`)

| File | Role |
|---|---|
| `glass.py` | Drop-in `GlassSettingCard`, `GlassSwitchSettingCard`, `GlassSettingCardGroup`/`GlassCard`, restyled switch, `GlassSlider`, value pill, `apply_widget_styles()`, `TextStyler`, tooltip style, `signals.appearanceChanged`. Same attribute API as the vendored widgets, so pages changed by import only. |
| `columns.py` | `TwoColumnBand`: consecutive groups are balanced into two columns from ~900 px of content, one column below. Groups are moved, never recreated. |
| `motion.py` | `Tween` (retargets from the shown value), `HoverProgress`, `fade_in()`. Animations off → every duration is 0. |
| `floating_nav.py` | `FloatingNav`: one accent capsule slides between items, active label expands, keyboard + hover + press states, `top`/`left` orientation, drag-to-dock. |
| `status_pill.py` | CPU %, quick preset slots 1–5, loaded preset; updated from the existing 1 s timer. |
| `caption_buttons.py`, `accent_picker.py`, `icon_utils.py` | Round caption buttons (the vendored buttons are kept for native maximise hit-testing), accent swatches + hex field, icon tinting. |

Row layout: `GlassSettingCard._fitHeight()` measures the wrapped description with font metrics at the real
text-column width and raises the row's minimum height. Qt's height-for-width through nested layouts
under-measured wide-control rows and cached stale heights, so rows deliberately return
`hasHeightForWidth() == False`. Sliders are flexible (96–320 px); the value pill gets a fixed width from the
widest text it can show.

### Tokens

- `ThemeColors.ACCENT/PRIMARY` follow `ui_accent.get_accent()` (`AccentColorPair`); `ui_accent.derive()`
  produces base / hi / lo / glow / soft / on-accent from one hex.
- `ThemeColors.GLASS_*` are the surface, border, hover, divider, pill, track, text and backdrop tokens,
  overridable under `"glass"` in `theme_colors.json`.
- New persisted settings (`_FIELD_MAP`, validated, preset/snapshot behaviour like `dark_mode`):
  `ui.accent_color` (default `#7c6cf0`), `ui.animations`, `ui.nav_dock` (`top|left`). Edited on
  Visuals → Appearance; Qt only.

### Motion

| Event | Behaviour |
|---|---|
| Page change | New page fades in (180 ms); rapid clicks retarget one tween, nothing queues. Fade, not blur — a blur effect on page-sized widgets was too costly. |
| Nav | Capsule slides/resizes in 260 ms; hover/press washes 120 ms. |
| Theme / accent | Snapshot overlay cross-fades out. Theme toggle skips the control restyle and the redundant `setThemeColor`. |
| Dock change | Content fades, orientation and margins apply, content fades back. |
| Constraint | A `QGraphicsEffect` is only attached while a fade runs; the stress test asserts none remain. |

### Quick preset slots

The bottom pill reads `config.preset_slots` (already used by Web Control). Right-clicking a preset on the
Configs page offers "Assign to quick slot N" / "Remove from quick slots". Clicking a slot runs
`ConfigManager.load_config()` and a light `_refreshAllPages(full=False)`.

## 4. Web Control client and PWA

Same token set as the Qt app: navy/charcoal glass, violet accent gradient, `data-theme` dark/light,
`data-motion` on/off (defaults to the OS reduced-motion setting), accent via inline `--accent*`. The tab bar is a
floating pill with a sliding capsule (icon-only and a bottom bar ≤ 760 px). The Studio and Instrument looks
are both kept. Class names and ids `app.js` relies on did not change, so the generic schema wiring, search
and visibility rules are untouched.

PWA: `manifest.webmanifest`, icons generated from the app logo, `sw.js` (shell-only precache;
network-first navigations with an `offline.html` fallback; **no `/api/*` or non-GET request is ever cached**),
install prompt, "Update available" banner (no automatic `skipWaiting`), correct headers via
`core/pwa_assets.static_headers()`. Asset version is bumped in five places (currently 26); `tests/test_pwa_assets.py`
checks they agree.

**Limit:** browsers only allow service workers and install in a secure context (`https://` or
`http://localhost`). Over plain `http://<LAN-ip>` — the default — the page works as before, but the
offline shell and install are unavailable; the Other tab says so. Built-in HTTPS was not implemented.

## 5. Testing and results

- `ruff check src tests tools`: clean.
- `pytest tests`: 1926 passed, 207 skipped, 4 failed. The 4 are the existing
  `test_screen_capture` failures (identical on the pre-redesign baseline). New: `test_ui_foundations.py`
  (35), `test_pwa_assets.py`, `test_slider_sync.py` updated to the new width constants.
- Offscreen Qt (`python tools/ui_render.py shots OUT` / `stress`; Python 3.12 + PyQt6 6.10, stub libEGL):
  every page rendered dark + light at 1280×820 and 900×600, plus left dock and a pink accent. The stress run
  covers 200 rapid nav clicks, 20 theme toggles, 20 accent changes, 20 dock switches, a width sweep
  900→1650→900 and animations off/on, and asserts no leaked graphics effects and no recreated widgets.
- Measured (offscreen, software raster, not representative of a GPU): theme toggle 4.2 s before → 1.4 s
  now; 200 rapid nav clicks 3.1–3.8 s; slider update median 0.7 ms.
- Chromium (Playwright) against a mock server wired to the real `web_control_settings`: capsule tracks the
  active tab, theme/accent/motion controls persist, manifest and icons valid, service worker registers on
  localhost, reload is controlled by it, shell cached, no `/api` cached, offline reload serves the shell,
  phone viewport has a bottom bar and no horizontal scroll, no console errors.

**Not verifiable here:** Windows acrylic/DWM, native snap layouts, Segoe fonts and real GPU smoothness. The
reference was compared by eye only.

## 6. Modified files

New: `src/core/{ui_accent,ui_motion,ui_layout,ui_status,pwa_assets}.py`;
`components/{glass,columns,motion,floating_nav,status_pill,caption_buttons,accent_picker,icon_utils}.py`;
`src/web_control_client/{manifest.webmanifest,sw.js,offline.html,icons/*}`;
`tests/{test_ui_foundations,test_pwa_assets}.py`; `tools/ui_render.py`; this file and `docs/ui_redesign/`.

Changed: `core/config.py`, `core/web_control_server.py`, `core/language_data/English_English.json`,
`window.py`, `base_page.py`, `theme_colors.py`, `slider_spin_card.py`, `no_wheel_widgets.py`, all ten pages
(`model`, `capture`, `inference`, `aim`, `trigger`, `keys`, `visuals`, `configs`, `convert`, `other`),
`web_control_client/{index.html,app.js,styles.css}`, `web_overlay/index.html` (meta only),
`tests/test_slider_sync.py`, `CLAUDE.md`.

## 7. Reverting to the classic look

The glass interface is the default. **Visuals → Appearance → Interface Style** offers *Glass* and *Classic*
(the interface as it was before the redesign: left navigation, stock setting cards, single column, fixed
blue accent). The choice is saved as `ui.style` and applied by restarting Axiom (a countdown notice; closing
it cancels the restart and the setting applies next launch). It is read once at start-up because pages bind
their widget classes at import time and own timers and threads; a live switch was rejected as too risky.
In Classic the accent colour, UI animations and navigation position settings do not apply and are hidden.
The style is excluded from config snapshots, so importing someone's file never changes your look. Code map:
`core/ui_style.py` (choice + metrics), `components/ui_kit.py` (widget classes per style),
`window_classic.py` (restored chrome), `components/restart_countdown.py`.
Offscreen renders of Classic match the pre-redesign renders; `tools/ui_render.py ... --style classic` reproduces them.

## 8. Remaining differences from the reference

- No wallpaper / background-image feature (declined); the backdrop is acrylic or a painted gradient.
- Docking is top and left only — no bottom or free-floating pill.
- Page change is a fade, not a blur.
- With ten pages only the active nav item shows its label in the Qt pill (the web tab bar shows all labels
  when there is room).
- The status pill has five preset slots (the reference shows three).
- Fonts follow the platform.
