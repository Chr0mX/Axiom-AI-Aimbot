// Axiom Web ESP — apexsky-style renderer.
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

  // ── Canvas sizing ──────────────────────────────────────────────
  function resize() {
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  // Map backend screen-space coords → canvas coords (handles browser viewport
  // differing from the game's render resolution).
  function scaler(s) {
    const sx = canvas.width / (s.screen.w || canvas.width);
    const sy = canvas.height / (s.screen.h || canvas.height);
    return { sx, sy };
  }

  // ── Colors (parity with overlay.py _BOX_THEMES) ────────────────
  const BOX_THEMES = {
    default: [0, 255, 140, 220],
    cyan: [0, 220, 255, 220],
    red: [255, 60, 60, 220],
    yellow: [255, 210, 0, 220],
    white: [255, 255, 255, 200],
    purple: [180, 60, 255, 210],
  };
  const rgba = (c) => `rgba(${c[0]},${c[1]},${c[2]},${(c[3] ?? 255) / 255})`;

  // ── WebSocket ──────────────────────────────────────────────────
  let ws = null;
  function connect() {
    const url = `ws://${location.hostname}:${WS_PORT}/`;
    ws = new WebSocket(url);
    ws.onopen = () => setStatus(true);
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
    statusEl.className = ok ? "connected" : "disconnected";
    statusEl.textContent = ok ? "● ESP" : "reconnecting…";
  }

  // ── Drawing primitives (parity with overlay.py paintEvent) ─────
  function cornerBox(x1, y1, x2, y2, color, thickness) {
    const w = x2 - x1, h = y2 - y1;
    const len = Math.max(6, Math.min(w, h) * 0.22);
    ctx.strokeStyle = rgba(color);
    ctx.lineWidth = thickness;
    ctx.beginPath();
    // TL
    ctx.moveTo(x1, y1 + len); ctx.lineTo(x1, y1); ctx.lineTo(x1 + len, y1);
    // TR
    ctx.moveTo(x2 - len, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + len);
    // BL
    ctx.moveTo(x1, y2 - len); ctx.lineTo(x1, y2); ctx.lineTo(x1 + len, y2);
    // BR
    ctx.moveTo(x2 - len, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - len);
    ctx.stroke();
  }

  function fovCorners(cx, cy, fov, color) {
    const half = fov / 2, len = Math.max(8, fov * 0.15);
    const x1 = cx - half, y1 = cy - half, x2 = cx + half, y2 = cy + half;
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
      // center / smart
      ty = (y1 + y2) * 0.5;
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

  // ── Render loop ────────────────────────────────────────────────
  function render() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    if (state && state.active !== false) {
      const st = state.settings;
      const { sx, sy } = scaler(state);
      const X = (v) => v * sx, Y = (v) => v * sy;
      const cx = X(state.center.x), cy = Y(state.center.y);

      // Detect range
      if (st.show_detect_range) {
        const rs = st.detect_range_size * sx;
        ctx.strokeStyle = "rgba(120,120,120,0.7)";
        ctx.lineWidth = 1;
        if (st.fov_circle_filter_enabled) {
          ctx.beginPath(); ctx.arc(cx, cy, rs / 2, 0, Math.PI * 2); ctx.stroke();
        } else {
          ctx.strokeRect(cx - rs / 2, cy - rs / 2, rs, rs);
        }
      }

      // FOV
      if (st.show_fov) {
        const fov = st.fov_size * sx;
        const fovColor = [255, 255, 255, 180];
        if (st.fov_circle_filter_enabled) {
          ctx.strokeStyle = rgba(fovColor); ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(cx, cy, fov / 2, 0, Math.PI * 2); ctx.stroke();
        } else {
          fovCorners(cx, cy, fov, fovColor);
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

          // in-FOV test in screen space
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

      // Crosshair
      if (st.show_crosshair) {
        const cc = st.crosshair_color, s = st.crosshair_size;
        ctx.strokeStyle = `rgb(${cc[0]},${cc[1]},${cc[2]})`;
        ctx.fillStyle = `rgb(${cc[0]},${cc[1]},${cc[2]})`;
        ctx.lineWidth = 1;
        if (st.crosshair_style === "cross") {
          ctx.beginPath();
          ctx.moveTo(cx - s, cy); ctx.lineTo(cx + s, cy);
          ctx.moveTo(cx, cy - s); ctx.lineTo(cx, cy + s);
          ctx.stroke();
        } else {
          ctx.beginPath(); ctx.arc(cx, cy, Math.max(1, s / 2), 0, Math.PI * 2); ctx.fill();
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

  function drawHud(s, count) {
    ctx.font = "11px Consolas, monospace";
    ctx.fillStyle = "rgba(0,224,160,0.85)";
    const lines = [`targets: ${count}`, `net: ${packetHz}hz`, `draw: ${drawFps}fps`];
    let y = canvas.height - 8 - (lines.length - 1) * 14;
    for (const ln of lines) { ctx.fillText(ln, 8, y); y += 14; }
  }

  function hsv2rgb(h, s, v) {
    const i = Math.floor(h * 6), f = h * 6 - i;
    const p = v * (1 - s), q = v * (1 - f * s), t = v * (1 - (1 - f) * s);
    let r, g, b;
    switch (i % 6) {
      case 0: r = v; g = t; b = p; break;
      case 1: r = q; g = v; b = p; break;
      case 2: r = p; g = v; b = t; break;
      case 3: r = p; g = q; b = v; break;
      case 4: r = t; g = p; b = v; break;
      default: r = v; g = p; b = q; break;
    }
    return [Math.round(r * 255), Math.round(g * 255), Math.round(b * 255)];
  }

  connect();
  requestAnimationFrame(render);
})();
