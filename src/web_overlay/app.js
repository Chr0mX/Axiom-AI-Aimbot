// Axiom Web HUD — apexsky-style renderer.
// The backend only streams state; this client owns all drawing. Network packets
// update `state`; a separate requestAnimationFrame loop always draws the newest
// state, so the overlay stays smooth even if packets arrive irregularly.

(() => {
  "use strict";

  const params = new URLSearchParams(location.search);
  const WS_PORT = params.get("ws") || "8765";

  const canvas = document.getElementById("esp");
  const ctx = canvas.getContext("2d");
  const statusEl = document.getElementById("status");

  // Latest decoded state (null until first packet).
  let state = null;
  // Render-side FPS + packet-rate tracking for the HUD.
  let lastPacketT = 0, packetHz = 0;
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
    hudX = Math.max(0, Math.min(canvas.width  - hudRect.w, e.clientX - dragOffX));
    hudY = Math.max(0, Math.min(canvas.height - hudRect.h, e.clientY - dragOffY));
  });
  window.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    canvas.classList.remove("dragging");
    savePos();
  });

  // ── Canvas sizing ──────────────────────────────────────────────
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  // Map backend screen-space coords → canvas coords
  function scaler(s) {
    const sx = canvas.width / (s.screen.w || canvas.width);
    const sy = canvas.height / (s.screen.h || canvas.height);
    return { sx, sy };
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
    ws.onopen  = () => setStatus(true);
    ws.onclose = () => { setStatus(false); setTimeout(connect, 1000); };
    ws.onerror = () => { try { ws.close(); } catch (e) {} };
    ws.onmessage = (ev) => {
      try {
        state = JSON.parse(ev.data);
        const now = performance.now();
        const dt = now - lastPacketT;
        if (lastPacketT && dt > 0) packetHz = Math.round(1000 / dt);
        lastPacketT = now;
      } catch (e) { /* ignore malformed */ }
    };
  }

  function setStatus(ok) {
    statusEl.className  = ok ? "connected" : "disconnected";
    statusEl.textContent = ok ? "● HUD" : "reconnecting…";
  }

  // ── Drawing primitives ─────────────────────────────────────────
  function cornerBox(x1, y1, x2, y2, color, thickness) {
    const w = x2 - x1, h = y2 - y1;
    const len = Math.max(6, Math.min(w, h) * 0.22);
    ctx.strokeStyle = rgba(color);
    ctx.lineWidth = thickness;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();
  }

  function fovCorners(cx, cy, fovX, fovY, color) {
    const halfX = fovX / 2, halfY = fovY / 2;
    const len = Math.max(8, Math.min(halfX, halfY) * 0.3);
    const x1 = cx - halfX, y1 = cy - halfY, x2 = cx + halfX, y2 = cy + halfY;
    ctx.strokeStyle = rgba(color);
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();
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
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state) {
      const st = state.settings;
      const { sx, sy } = scaler(state);
      const X = (v) => v * sx, Y = (v) => v * sy;
      const cx = X(state.center.x), cy = Y(state.center.y);

      // Detect range — sx/sy scaled independently, since screen.w/h (e.g. a
      // square UDP stream) rarely matches the canvas's own aspect ratio
      // (the browser viewport); a single scalar stretches this into an oval.
      if (st.show_detect_range) {
        const rsx = st.detect_range_size * sx, rsy = st.detect_range_size * sy;
        ctx.strokeStyle = "rgba(120,120,120,0.7)";
        ctx.lineWidth = 1;
        if (st.fov_circle_filter_enabled) {
          ctx.beginPath(); ctx.ellipse(cx, cy, rsx / 2, rsy / 2, 0, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeRect(cx - rsx / 2, cy - rsy / 2, rsx, rsy);
        }
      }

      // FOV — same per-axis scaling as detect range, for the same reason.
      if (st.show_fov) {
        const fovx = st.fov_size * sx, fovy = st.fov_size * sy;
        const fovColor = [255, 255, 255, 180];
        if (st.fov_circle_filter_enabled) {
          ctx.strokeStyle = rgba(fovColor); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.ellipse(cx, cy, fovx / 2, fovy / 2, 0, 0, Math.PI * 2); ctx.stroke();
        } else {
          fovCorners(cx, cy, fovx, fovy, fovColor);
        }
      }

      // Boxes
      const boxes = state.boxes || [];
      const confs = state.confidences || [];
      if (st.show_boxes) {
        const theme = BOX_THEMES[(st.box_color_theme || "default").toLowerCase()] || BOX_THEMES.default;
        const useChroma = st.chroma_box_speed > 0;
        const hue = useChroma ? (performance.now() / 1000 * st.chroma_box_speed * 60) % 360 : 0;
        const fovHalf = st.fov_size / 2;

        for (let i = 0; i < boxes.length; i++) {
          const b = boxes[i];
          const x1 = X(b[0]), y1 = Y(b[1]), x2 = X(b[2]), y2 = Y(b[3]);
          const conf = i < confs.length ? confs[i] : 0.5;
          const thickness = Math.max(2, Math.min(4, 2 + Math.round(conf * 2)));

          const ox = state.center.x, oy = state.center.y;
          let inFov;
          if (st.fov_circle_filter_enabled) {
            const nx = Math.min(Math.max(ox, b[0]), b[2]);
            const ny = Math.min(Math.max(oy, b[1]), b[3]);
            inFov = (nx - ox) ** 2 + (ny - oy) ** 2 <= fovHalf * fovHalf;
          } else {
            inFov = (b[0] < ox + fovHalf && b[2] > ox - fovHalf &&
                     b[1] < oy + fovHalf && b[3] > oy - fovHalf);
          }

          let color = theme;
          if (useChroma && inFov) {
            const c = hsv2rgb(hue / 360, 1, 1); color = [c[0], c[1], c[2], 220];
          }

          if (st.box_full_rect) {
            ctx.strokeStyle = rgba(color); ctx.lineWidth = thickness;
            ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
          } else {
            cornerBox(x1, y1, x2, y2, color, thickness);
          }

          if (st.show_confidence && i < confs.length) {
            ctx.fillStyle = "rgba(255,255,255,0.9)";
            ctx.font = "bold 12px Arial";
            ctx.fillText(`${Math.round(conf * 100)}%`, x1 - 4, y1 - 6);
          }

          const [tx, ty] = aimPoint(x1, y1, x2, y2, st);
          if (st.show_tracer_line) {
            ctx.strokeStyle = "rgba(0,220,255,0.5)"; ctx.lineWidth = 1;
            ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(tx, ty); ctx.stroke();
          }
        }
      }

      // Locked-box highlight
      if (state.locked_box) {
        const lb = state.locked_box;
        ctx.strokeStyle = state.locked_decaying ? "rgba(255,170,0,0.9)" : "rgba(0,255,140,0.95)";
        ctx.lineWidth = 2;
        ctx.strokeRect(X(lb[0]), Y(lb[1]), X(lb[2]) - X(lb[0]), Y(lb[3]) - Y(lb[1]));
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
    }

    // FPS counter
    frames++;
    const now = performance.now();
    if (now - fpsT >= 500) { drawFps = Math.round((frames * 1000) / (now - fpsT)); frames = 0; fpsT = now; }

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

    const aim = s.active;
    const firing = s.aim_firing;
    const model = (s.model || "").replace(/\.onnx$/i, "") || "none";
    const capMethod = (s.screenshot_method || "dxcam").toUpperCase();
    const capFps = s.capture_fps   != null ? `${s.capture_fps.toFixed(1)} fps`   : "—";
    const infFps = s.inference_fps != null ? `${s.inference_fps.toFixed(1)} fps` : "—";
    const latStr = latencyMs != null ? `${latencyMs}ms` : "—";

    const lines = [
      { text: aim ? "● Aim ON" : "● Aim OFF",
        color: aim ? GREEN : RED },
      { text: `Aim Status  ${firing ? "● Active" : "● Idle"}`,
        color: firing ? GREEN : RED },
      { text: `Model: ${model}`,                    color: NEUTRAL },
      { text: `Capture Method: ${capMethod}`,        color: NEUTRAL },
      { text: `Capture FPS: ${capFps}`,              color: NEUTRAL },
      { text: `Inference FPS: ${infFps}`,            color: NEUTRAL },
      { text: `Target: ${count}`,                    color: "rgba(0,224,160,0.88)" },
      { text: `Net Latency: ${latStr}  |  Draw FPS: ${drawFps}`,
        color: "rgba(110,110,130,0.7)" },
    ];

    // Measure widest line for background rect
    let maxW = 0;
    for (const ln of lines) maxW = Math.max(maxW, ctx.measureText(ln.text).width);

    const boxW = maxW + pad * 2;
    const boxH = lines.length * lh + pad * 2 - (lh - fs);

    // Clamp to viewport
    hudX = Math.max(0, Math.min(canvas.width  - boxW, hudX));
    hudY = Math.max(0, Math.min(canvas.height - boxH, hudY));

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
