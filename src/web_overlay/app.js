// Axiom Web HUD — apexsky-style renderer.
// The backend only streams state; this client owns all drawing. Network packets
// update `state`; a separate requestAnimationFrame loop always draws the newest
// state, so the overlay stays smooth even if packets arrive irregularly.

(() => {
  "use strict";

  const params = new URLSearchParams(location.search);
  const WS_PORT = params.get("ws") || "8765";

  const canvas = document.getElementById("esp");
  // desynchronized: ask the browser to present the canvas without waiting for the compositor
  // (lower latency on Chromium; ignored elsewhere). alpha stays on: the page is transparent.
  const ctx = canvas.getContext("2d", { desynchronized: true });
  const statusEl = document.getElementById("status");

  // Latest decoded state (null until first packet).
  let state = null;
  // Render-side FPS + packet-rate tracking for the HUD.
  // Newest undecoded packet (JSON text). Decoding happens once per drawn frame in render(), so
  // a burst of packets between two frames costs one JSON.parse, not one each.
  let pendingRaw = null, packets = 0, packetsT = performance.now(), dataHz = 0;
  let frames = 0, fpsT = performance.now(), drawFps = 0;
  // 2PC latency (browser → HTTP server round-trip)
  let latencyMs = null;
  async function measureLatency() {
    try {
      const t0 = performance.now();
      await fetch("/ping", { cache: "no-store" });
      latencyMs = Math.round(performance.now() - t0);
    } catch { latencyMs = null; }
  }
  measureLatency();
  setInterval(measureLatency, 2000);

  // ── HUD settings (persisted to localStorage) ──────────────────
  const CFG_DEFAULTS = {
    fontSize:      11,
    bgColor:       "#000000",
    bgOpacity:     65,
    borderColor:   "#00e0a0",
    borderOpacity: 60,
    borderWidth:   1,
    borderRadius:  4,
    padding:       6,
    // Detection boxes. "app"/0 = follow the app's own setting; anything
    // else overrides it for this browser only (the Web ESP server is
    // read-only, so it can't change the app's config).
    boxStyle:      "app",   // "app" | "corners" | "full"
    boxThickness:  0,       // px, 0 = app
    boxCornerPct:  0,       // % of shorter side, 0 = app
  };
  const LS_CFG = "axiom_hud_cfg";
  const LS_POS = "axiom_hud_pos";

  function loadCfg() {
    try { return Object.assign({}, CFG_DEFAULTS, JSON.parse(localStorage.getItem(LS_CFG) || "{}")); }
    catch { return Object.assign({}, CFG_DEFAULTS); }
  }
  function saveCfg() { localStorage.setItem(LS_CFG, JSON.stringify(cfg)); }

  let cfg = loadCfg();

  // HUD position (drag-repositionable)
  let hudX = 8, hudY = 8;
  try {
    const p = JSON.parse(localStorage.getItem(LS_POS) || "{}");
    if (typeof p.x === "number") hudX = p.x;
    if (typeof p.y === "number") hudY = p.y;
  } catch {}

  function savePos() { localStorage.setItem(LS_POS, JSON.stringify({ x: hudX, y: hudY })); }

  // Last drawn HUD bounding box (for drag hit-test)
  let hudRect = { x: 0, y: 0, w: 0, h: 0 };

  // ── Drag state ─────────────────────────────────────────────────
  let dragging = false, dragOffX = 0, dragOffY = 0;

  canvas.addEventListener("mousedown", (e) => {
    const r = hudRect;
    if (e.clientX >= r.x && e.clientX <= r.x + r.w &&
        e.clientY >= r.y && e.clientY <= r.y + r.h) {
      dragging = true;
      dragOffX = e.clientX - hudX;
      dragOffY = e.clientY - hudY;
      canvas.classList.add("dragging");
      e.preventDefault();
    }
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    hudX = Math.max(0, Math.min(view.w - hudRect.w, e.clientX - dragOffX));
    hudY = Math.max(0, Math.min(view.h - hudRect.h, e.clientY - dragOffY));
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove("dragging");
    savePos();
  });

  // ── Canvas sizing ──────────────────────────────────────────────
  // `view` is the drawable size in CSS px; the backing store is dpr times larger (capped at 2) so
  // lines stay sharp on HiDPI screens without an expensive 3x canvas. Everything draws in CSS px.
  let view = { w: window.innerWidth, h: window.innerHeight, dpr: 1 };
  function resize() {
    const dpr = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    view = { w: window.innerWidth, h: window.innerHeight, dpr };
    canvas.width = Math.round(view.w * dpr);
    canvas.height = Math.round(view.h * dpr);
    canvas.style.width = view.w + "px";
    canvas.style.height = view.h + "px";
  }
  window.addEventListener("resize", resize);
  resize();

  // Map backend screen-space coords → canvas coords.
  //
  // Uses a single uniform scale, not independent per-axis sx/sy — the
  // browser viewport's aspect ratio essentially never matches the
  // captured screen/game resolution's (window.innerWidth/innerHeight can
  // be any shape the user's browser happens to be), so scaling each axis
  // separately to fill the canvas stretches every square shape (the FOV
  // box, the detect-range box, roughly-square target boxes) into a
  // rectangle, and inflates them disproportionately along whichever axis
  // the mismatch favors. A uniform scale — the smaller of the two
  // per-axis ratios, i.e. "contain" — keeps every shape's real aspect
  // ratio intact; offX/offY letterbox (center) the result within the
  // canvas instead of stretching it to fill.
  function scaler(s) {
    const screenW = s.screen.w || view.w;
    const screenH = s.screen.h || view.h;
    const scale = Math.min(view.w / screenW, view.h / screenH) || 1;
    const offX = (view.w - screenW * scale) / 2;
    const offY = (view.h - screenH * scale) / 2;
    return { sx: scale, sy: scale, offX, offY };
  }

  // ── Colors ─────────────────────────────────────────────────────
  const BOX_THEMES = {
    default: [0, 255, 140, 220],
    cyan:    [0, 220, 255, 220],
    red:     [255, 60,  60,  220],
    yellow:  [255, 210, 0,   220],
    white:   [255, 255, 255, 200],
    purple:  [180, 60,  255, 210],
  };
  const rgba = (c) => `rgba(${c[0]},${c[1]},${c[2]},${(c[3] ?? 255) / 255})`;

  function hexToRgb(hex) {
    const v = parseInt(hex.slice(1), 16);
    return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
  }

  // ── WebSocket ──────────────────────────────────────────────────
  let ws = null;
  function connect() {
    const url = `ws://${location.hostname}:${WS_PORT}/`;
    ws = new WebSocket(url);
    ws.onopen  = () => { connected = true; badgeKey = null; };
    ws.onclose = () => { connected = false; setStatus(false); setTimeout(connect, 1000); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
    ws.onmessage = (ev) => {
      pendingRaw = ev.data;      // keep only the newest; decoded in render()
      packets++;
    };
  }

  // Top-right badge: connection dot + the Visuals overlays that are switched on, in the same
  // order as the Visuals page. Wording matches the Web Control client (connecting… / disconnected).
  let connected = false, badgeKey = null;
  const BADGE_ITEMS = [
    ["show_crosshair", "Crosshair"],
    ["show_fov", "FOV"],
    ["show_boxes", "Boxes"],
    ["show_confidence", "Confidence"],
    ["show_detect_range", "Detection Range"],
    ["show_tracer_line", "Tracer Line"],
    ["show_aim_prediction_marker", "Predicted Aim Point"],
  ];
  function enabledOverlays(st) {
    return BADGE_ITEMS.filter(([key]) => !!(st && st[key])).map(([, label]) => label);
  }
  function setStatus(ok, st) {
    const text = ok ? "\u25cf " + (enabledOverlays(st).join(" \u00b7 ") || "connected") : "disconnected";
    const key = (ok ? "1" : "0") + text;
    if (key === badgeKey) return;             // don't touch the DOM every packet
    badgeKey = key;
    statusEl.className  = ok ? "connected" : "disconnected";
    statusEl.textContent = text;
  }

  // ── Drawing primitives ─────────────────────────────────────────
  // Mirrors core/box_style.py: a fixed thickness / corner % when set,
  // otherwise this page's original automatic look.
  function boxThicknessFor(st, conf) {
    const fixed = cfg.boxThickness > 0 ? cfg.boxThickness : (st.box_line_thickness || 0);
    if (fixed > 0) return Math.min(10, fixed);
    return Math.max(2, Math.min(4, 2 + Math.round(conf * 2)));
  }
  function cornerLenFor(st, w, h) {
    const side = Math.max(0, Math.min(Math.abs(w), Math.abs(h)));
    const pct = cfg.boxCornerPct > 0 ? cfg.boxCornerPct : (st.box_corner_length_pct || 0);
    if (pct > 0) return Math.max(2, Math.min(side / 2, side * Math.min(50, pct) / 100));
    return Math.max(6, side * 0.22);
  }
  function lockedBoxThickness(st) {
    const fixed = cfg.boxThickness > 0 ? cfg.boxThickness : (st.box_line_thickness || 0);
    return fixed > 0 ? Math.min(10, fixed) : 2;
  }
  function useFullRect(st) {
    if (cfg.boxStyle === "full") return true;
    if (cfg.boxStyle === "corners") return false;
    return !!st.box_full_rect;
  }

  function cornerBox(x1, y1, x2, y2, color, thickness, len) {
    ctx.strokeStyle = rgba(color);
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();
  }

  function fovCorners(cx, cy, fovX, fovY, color, width) {
    const halfX = fovX / 2, halfY = fovY / 2;
    const len = Math.max(8, Math.min(halfX, halfY) * 0.3);
    const x1 = cx - halfX, y1 = cy - halfY, x2 = cx + halfX, y2 = cy + halfY;
    ctx.strokeStyle = rgba(color);
    ctx.lineWidth = width || 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();
  }

  // True ellipse/AABB intersection test (semi-axes a, b) — mirrors
  // ai_loop_utils._ellipse_intersects_bbox exactly: scale the ellipse
  // centre and the box corners by (1/a, 1/b), turning the ellipse into a
  // unit circle and the box into an axis-aligned box in the same
  // transformed space, then run the usual closest-point-on-box test there.
  function ellipseIntersectsBbox(cx, cy, a, b, x1, y1, x2, y2) {
    if (a <= 0 || b <= 0) return false;
    const lx = Math.min(x1, x2), rx = Math.max(x1, x2);
    const ty = Math.min(y1, y2), by = Math.max(y1, y2);
    const tcx = cx / a, tcy = cy / b;
    const tlx = lx / a, trx = rx / a;
    const tty = ty / b, tby = by / b;
    const nx = Math.min(Math.max(tcx, tlx), trx);
    const ny = Math.min(Math.max(tcy, tty), tby);
    return (nx - tcx) ** 2 + (ny - tcy) ** 2 <= 1.0;
  }

  function aimPoint(x1, y1, x2, y2, st) {
    const w = x2 - x1, h = y2 - y1;
    const tx = x1 + w * 0.5;
    let ty;
    if (st.aim_part === "head") {
      ty = y1 + h * st.head_height_ratio * 0.5;
    } else if (st.aim_part === "body") {
      ty = (y1 + h * st.head_height_ratio + y2) * 0.5;
    } else if (st.aim_part === "custom") {
      ty = y1 + h * ((st.aim_custom_y_pct ?? 30) / 100);
    } else {
      // center / smart — uses custom Y offset
      ty = y1 + h * ((st.aim_custom_y_pct ?? 50) / 100);
    }
    const r = Math.max(3, Math.min(6, w / 8));
    ctx.strokeStyle = "rgba(255,80,80,0.86)";
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    ctx.moveTo(tx - r, ty - r); ctx.lineTo(tx + r, ty + r);
    ctx.moveTo(tx + r, ty - r); ctx.lineTo(tx - r, ty + r);
    ctx.stroke();
    return [tx, ty];
  }

  // ── Rounded rect helper ────────────────────────────────────────
  function roundRect(x, y, w, h, r) {
    r = Math.min(r, w / 2, h / 2);
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.arcTo(x + w, y,     x + w, y + r,     r);
    ctx.lineTo(x + w, y + h - r);
    ctx.arcTo(x + w, y + h, x + w - r, y + h, r);
    ctx.lineTo(x + r, y + h);
    ctx.arcTo(x,     y + h, x,     y + h - r, r);
    ctx.lineTo(x,     y + r);
    ctx.arcTo(x,     y,     x + r, y,         r);
    ctx.closePath();
  }

  // ── Render loop ────────────────────────────────────────────────
  function render() {
    if (pendingRaw !== null) {
      try { state = JSON.parse(pendingRaw); } catch (e) { /* ignore malformed */ }
      pendingRaw = null;
    }
    ctx.setTransform(view.dpr, 0, 0, view.dpr, 0, 0);
    ctx.clearRect(0, 0, view.w, view.h);
    if (connected) setStatus(true, state && state.settings);

    if (state) {
      const st = state.settings;
      const { sx, sy, offX, offY } = scaler(state);
      const X = (v) => v * sx + offX, Y = (v) => v * sy + offY;
      const cx = X(state.center.x), cy = Y(state.center.y);

      // Detect range — sx === sy (see scaler()), so this stays a true
      // square/circle regardless of the browser window's own aspect ratio.
      if (st.show_detect_range) {
        const rsx = st.detect_range_size * sx, rsy = st.detect_range_size * sy;
        ctx.strokeStyle = "rgba(120,120,120,0.7)";
        ctx.lineWidth = st.detect_range_line_thickness || 1;
        if (st.fov_circle_filter_enabled) {
          ctx.beginPath(); ctx.ellipse(cx, cy, rsx / 2, rsy / 2, 0, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeRect(cx - rsx / 2, cy - rsy / 2, rsx, rsy);
        }
      }

      // FOV — fov_height defaults to fov_size server-side when a config
      // predates it, so this is a square unless the user has set them
      // independently (see model_page... no — inference_page.py's FOV
      // Width/Height sliders).
      if (st.show_fov) {
        const fovHeight = st.fov_height != null ? st.fov_height : st.fov_size;
        const fovx = st.fov_size * sx, fovy = fovHeight * sy;
        const fovColor = [255, 255, 255, 180];
        if (st.fov_circle_filter_enabled) {
          ctx.strokeStyle = rgba(fovColor); ctx.lineWidth = st.fov_line_thickness || 2;
          ctx.beginPath(); ctx.ellipse(cx, cy, fovx / 2, fovy / 2, 0, 0, Math.PI * 2); ctx.stroke();
        } else {
          fovCorners(cx, cy, fovx, fovy, fovColor, st.fov_line_thickness || 2);
        }
      }

      // Retention FOV — where a locked target is still tracked (not aimed) after
      // the FOV shrinks past it. Only sent while "Keep Lock Outside Shrunk FOV"
      // is active; drawn dashed so it reads as a boundary, not another FOV.
      if (st.show_fov && st.retention_w > 0) {
        const rx = st.retention_w * sx, ry = (st.retention_h || st.retention_w) * sy;
        ctx.save();
        ctx.setLineDash([6, 6]);
        ctx.strokeStyle = state.locked_track_only ? "rgba(0,200,255,0.75)" : "rgba(255,255,255,0.28)";
        ctx.lineWidth = 1;
        if (st.fov_circle_filter_enabled) {
          ctx.beginPath(); ctx.ellipse(cx, cy, rx / 2, ry / 2, 0, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeRect(cx - rx / 2, cy - ry / 2, rx, ry);
        }
        ctx.restore();
      }

      // Boxes
      const boxes = state.boxes || [];
      const confs = state.confidences || [];
      if (st.show_boxes) {
        const theme = BOX_THEMES[(st.box_color_theme || "default").toLowerCase()] || BOX_THEMES.default;
        const useChroma = st.chroma_box_speed > 0;
        const hue = useChroma ? (performance.now() / 1000 * st.chroma_box_speed * 60) % 360 : 0;
        const fovHalfX = st.fov_size / 2;
        const fovHalfY = (st.fov_height != null ? st.fov_height : st.fov_size) / 2;

        for (let i = 0; i < boxes.length; i++) {
          const b = boxes[i];
          const x1 = X(b[0]), y1 = Y(b[1]), x2 = X(b[2]), y2 = Y(b[3]);
          const conf = i < confs.length ? confs[i] : 0.5;
          const thickness = boxThicknessFor(st, conf);

          const ox = state.center.x, oy = state.center.y;
          let inFov;
          if (st.fov_circle_filter_enabled) {
            // Same ellipse/AABB test as ai_loop_utils._ellipse_intersects_bbox
            // (scale into a space where the ellipse is a unit circle, then
            // the usual closest-point-on-box test) — a circle when
            // fovHalfX == fovHalfY, same as it always was.
            inFov = ellipseIntersectsBbox(ox, oy, fovHalfX, fovHalfY, b[0], b[1], b[2], b[3]);
          } else {
            inFov = (b[0] < ox + fovHalfX && b[2] > ox - fovHalfX &&
                     b[1] < oy + fovHalfY && b[3] > oy - fovHalfY);
          }

          let color = theme;
          if (useChroma && inFov) {
            const c = hsv2rgb(hue / 360, 1, 1); color = [c[0], c[1], c[2], 220];
          }

          if (useFullRect(st)) {
            ctx.strokeStyle = rgba(color); ctx.lineWidth = thickness;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          } else {
            cornerBox(x1, y1, x2, y2, color, thickness, cornerLenFor(st, x2 - x1, y2 - y1));
          }

          if (st.show_confidence && i < confs.length) {
            ctx.fillStyle = "rgba(255,255,255,0.9)";
            ctx.font = "bold 12px Arial";
            ctx.fillText(`${Math.round(conf * 100)}%`, x1 - 4, y1 - 6);
          }

          if (st.show_tracer_line && inFov) {
            const [tx, ty] = aimPoint(x1, y1, x2, y2, st);
            ctx.strokeStyle = "rgba(0,220,255,0.5)"; ctx.lineWidth = st.tracer_line_thickness || 2;
            ctx.beginPath();
            if (st.tracer_from_bottom) ctx.moveTo(X(state.screen.w / 2), Y(state.screen.h));   // bottom-center of the screen
            else ctx.moveTo(cx, cy);
            ctx.lineTo(tx, ty); ctx.stroke();
          }
        }
      }

      // Locked-box highlight
      if (state.locked_box) {
        const lb = state.locked_box;
        // amber = decaying (target missing), cyan dashed = tracked outside the
        // control FOV (kept, not aimed), green = actively locked.
        ctx.strokeStyle = state.locked_decaying ? "rgba(255,170,0,0.9)"
          : state.locked_track_only ? "rgba(0,200,255,0.95)" : "rgba(0,255,140,0.95)";
        ctx.setLineDash(state.locked_track_only ? [5, 4] : []);
        // Follows the box thickness setting too: this outline sits on top of
        // the locked target's own box, so a fixed 2 px here hid any change.
        ctx.lineWidth = lockedBoxThickness(state.settings || {});
        ctx.strokeRect(X(lb[0]), Y(lb[1]), X(lb[2]) - X(lb[0]), Y(lb[3]) - Y(lb[1]));
        ctx.setLineDash([]);
      }

      // Predicted aim point — hollow circle at the locked target's post-prediction/Kalman point
      // (the in-game overlay's marker). Sent only while a lock exists and prediction is on.
      if (st.show_aim_prediction_marker && state.aim_predicted) {
        const pr = Math.max(4, 7 * sx);
        ctx.strokeStyle = "rgba(80,210,255,0.92)";
        ctx.lineWidth = st.aim_marker_line_thickness || 2;
        ctx.beginPath();
        ctx.ellipse(X(state.aim_predicted.x), Y(state.aim_predicted.y), pr, pr, 0, 0, Math.PI * 2);
        ctx.stroke();
      }

      // Crosshair — scaled per axis like everything else above; previously
      // drawn at the raw crosshair_size with no scaling at all.
      if (st.show_crosshair) {
        const cc = st.crosshair_color;
        const sxr = st.crosshair_size * sx, syr = st.crosshair_size * sy;
        ctx.strokeStyle = `rgb(${cc[0]},${cc[1]},${cc[2]})`;
        ctx.fillStyle   = `rgb(${cc[0]},${cc[1]},${cc[2]})`;
        ctx.lineWidth = 1;
        if (st.crosshair_style === "cross") {
          ctx.beginPath();
          ctx.moveTo(cx - sxr, cy); ctx.lineTo(cx + sxr, cy);
          ctx.moveTo(cx, cy - syr); ctx.lineTo(cx, cy + syr);
          ctx.stroke();
        } else {
          ctx.beginPath();
          ctx.ellipse(cx, cy, Math.max(1, sxr / 2), Math.max(1, syr / 2), 0, 0, Math.PI * 2);
          ctx.fill();
        }
      }

      drawHud(state, boxes.length);
    } else {
      drawHud(null, 0);          // placeholder rows ("—") until the first packet, like Web Control
    }

    // Draw FPS (requestAnimationFrame rate: capped by the display's refresh rate) and the rate
    // data packets actually arrive at -- two different numbers.
    frames++;
    const now = performance.now();
    if (now - fpsT >= 500) { drawFps = Math.round((frames * 1000) / (now - fpsT)); frames = 0; fpsT = now; }
    if (now - packetsT >= 500) { dataHz = Math.round((packets * 1000) / (now - packetsT)); packets = 0; packetsT = now; }

    requestAnimationFrame(render);
  }

  // ── HUD draw ───────────────────────────────────────────────────
  function drawHud(s, count) {
    const fs = cfg.fontSize;
    const lh = Math.round(fs * 1.35);
    const pad = cfg.padding;

    ctx.font = `${fs}px Consolas, monospace`;

    const GREEN = "rgba(0,255,100,0.92)";
    const RED   = "rgba(255,60,60,0.9)";
    const NEUTRAL = "rgba(200,200,214,0.82)";

    // Wording matches the Web Control sidebar: no bullets, no unit suffixes, "\u2014" when empty.
    const has = !!s;
    const aim = has && s.active;
    const firing = has && s.aim_firing;
    const dash = "\u2014";
    const model = has && s.model ? s.model.replace(/\.onnx$/i, "") : dash;
    const capMethod = has && s.screenshot_method ? s.screenshot_method.toUpperCase() : dash;
    const num = (v) => (v != null && isFinite(v) ? Number(v).toFixed(1) : dash);
    const latStr = latencyMs != null ? `${latencyMs}ms` : dash;

    const lines = [
      { text: `Aim Toggle: ${has ? (aim ? "ON" : "OFF") : dash}`,
        color: !has ? NEUTRAL : aim ? GREEN : RED },
      { text: `Aim Status: ${has ? (firing ? "Aiming" : "Idle") : dash}`,
        color: !has ? NEUTRAL : firing ? GREEN : RED },
      { text: `Model: ${model}`,                              color: NEUTRAL },
      { text: `Capture Method: ${capMethod}`,                 color: NEUTRAL },
      { text: `Capture FPS: ${has ? num(s.capture_fps) : dash}`,     color: NEUTRAL },
      { text: `Inference FPS: ${has ? num(s.inference_fps) : dash}`, color: NEUTRAL },
      { text: `Target: ${has ? count : dash}`,                color: "rgba(0,224,160,0.88)" },
      { text: `Net Latency: ${latStr}  |  Draw FPS: ${drawFps}  |  Data: ${has ? dataHz + " Hz" : dash}`,
        color: "rgba(110,110,130,0.7)" },
    ];

    // Measure widest line for background rect
    let maxW = 0;
    for (const ln of lines) maxW = Math.max(maxW, ctx.measureText(ln.text).width);

    const boxW = maxW + pad * 2;
    const boxH = lines.length * lh + pad * 2 - (lh - fs);

    // Clamp to viewport
    hudX = Math.max(0, Math.min(view.w - boxW, hudX));
    hudY = Math.max(0, Math.min(view.h - boxH, hudY));

    // Update hit-test rect
    hudRect = { x: hudX, y: hudY, w: boxW, h: boxH };

    const rr = cfg.borderRadius;

    // Background
    const [br, bg, bb] = hexToRgb(cfg.bgColor);
    ctx.fillStyle = `rgba(${br},${bg},${bb},${cfg.bgOpacity / 100})`;
    roundRect(hudX, hudY, boxW, boxH, rr);
    ctx.fill();

    // Border
    if (cfg.borderWidth > 0) {
      const [er, eg, eb] = hexToRgb(cfg.borderColor);
      ctx.strokeStyle = `rgba(${er},${eg},${eb},${cfg.borderOpacity / 100})`;
      ctx.lineWidth = cfg.borderWidth;
      roundRect(hudX, hudY, boxW, boxH, rr);
      ctx.stroke();
    }

    // Text
    const tx = hudX + pad;
    let ty = hudY + pad + fs;
    for (const ln of lines) {
      ctx.fillStyle = ln.color;
      ctx.fillText(ln.text, tx, ty);
      ty += lh;
    }
  }

  // ── Settings panel wiring ──────────────────────────────────────
  const cfgToggle = document.getElementById("cfg-toggle");
  const cfgPanel  = document.getElementById("cfg-panel");

  cfgToggle.addEventListener("click", () => cfgPanel.classList.toggle("open"));

  // Close panel when clicking outside
  document.addEventListener("mousedown", (e) => {
    if (!cfgPanel.contains(e.target) && e.target !== cfgToggle)
      cfgPanel.classList.remove("open");
  });

  // ── Gear auto-fade: hide after 5s idle, reappear on any activity/hover ──
  let idleTimer = null;
  function scheduleGearFade() {
    cfgToggle.classList.remove("faded");
    clearTimeout(idleTimer);
    idleTimer = setTimeout(() => {
      if (!cfgPanel.classList.contains("open")) cfgToggle.classList.add("faded");
    }, 5000);
  }
  window.addEventListener("mousemove", scheduleGearFade);
  window.addEventListener("mousedown", scheduleGearFade);
  cfgToggle.addEventListener("mouseenter", scheduleGearFade);
  scheduleGearFade();

  function bindRange(id, key, valId) {
    const el = document.getElementById(id);
    const vl = document.getElementById(valId);
    el.value = cfg[key];
    vl.textContent = cfg[key];
    el.addEventListener("input", () => {
      cfg[key] = Number(el.value);
      vl.textContent = el.value;
      saveCfg();
    });
  }

  function bindColor(id, key) {
    const el = document.getElementById(id);
    el.value = cfg[key];
    el.addEventListener("input", () => {
      cfg[key] = el.value;
      saveCfg();
    });
  }

  bindRange("cfg-font-size",      "fontSize",      "val-font-size");
  bindColor("cfg-bg-color",       "bgColor");
  bindRange("cfg-bg-opacity",     "bgOpacity",     "val-bg-opacity");
  bindColor("cfg-border-color",   "borderColor");
  bindRange("cfg-border-opacity", "borderOpacity", "val-border-opacity");
  bindRange("cfg-border-width",   "borderWidth",   "val-border-width");
  bindRange("cfg-border-radius",  "borderRadius",  "val-border-radius");
  bindRange("cfg-padding",        "padding",       "val-padding");

  // Box overrides — 0 shows as "App" (follow the app's setting).
  function bindBoxRange(id, key, valId, unit) {
    const el = document.getElementById(id);
    const vl = document.getElementById(valId);
    const show = () => { vl.textContent = cfg[key] > 0 ? cfg[key] + unit : "App"; };
    el.value = cfg[key];
    show();
    el.addEventListener("input", () => { cfg[key] = Number(el.value); show(); saveCfg(); });
    return show;
  }
  const showBoxThickness = bindBoxRange("cfg-box-thickness", "boxThickness", "val-box-thickness", "px");
  const showBoxCorner    = bindBoxRange("cfg-box-corner",    "boxCornerPct", "val-box-corner",    "%");
  const boxStyleEl = document.getElementById("cfg-box-style");
  boxStyleEl.value = cfg.boxStyle;
  boxStyleEl.addEventListener("change", () => { cfg.boxStyle = boxStyleEl.value; saveCfg(); });

  document.getElementById("cfg-reset").addEventListener("click", () => {
    cfg = Object.assign({}, CFG_DEFAULTS);
    saveCfg();
    // Re-sync all inputs
    [
      ["cfg-font-size",      "fontSize",      "val-font-size"],
      ["cfg-bg-opacity",     "bgOpacity",     "val-bg-opacity"],
      ["cfg-border-opacity", "borderOpacity", "val-border-opacity"],
      ["cfg-border-width",   "borderWidth",   "val-border-width"],
      ["cfg-border-radius",  "borderRadius",  "val-border-radius"],
      ["cfg-padding",        "padding",       "val-padding"],
    ].forEach(([id, key, vid]) => {
      document.getElementById(id).value = cfg[key];
      document.getElementById(vid).textContent = cfg[key];
    });
    document.getElementById("cfg-bg-color").value     = cfg.bgColor;
    document.getElementById("cfg-border-color").value = cfg.borderColor;
    document.getElementById("cfg-box-thickness").value = cfg.boxThickness;
    document.getElementById("cfg-box-corner").value = cfg.boxCornerPct;
    boxStyleEl.value = cfg.boxStyle;
    showBoxThickness();
    showBoxCorner();
  });

  // ── HSV helper ─────────────────────────────────────────────────
  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v*(1-s), q = v*(1-f*s), t = v*(1-(1-f)*s);
    let r, g, b;
    switch (i % 6) {
      case 0: r=v; g=t; b=p; break;
      case 1: r=q; g=v; b=p; break;
      case 2: r=p; g=v; b=t; break;
      case 3: r=p; g=q; b=v; break;
      case 4: r=t; g=p; b=v; break;
      default: r=v; g=p; b=q; break;
    }
    return [Math.round(r*255), Math.round(g*255), Math.round(b*255)];
  }

  connect();
  requestAnimationFrame(render);
})();
