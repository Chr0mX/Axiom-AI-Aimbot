(function () {
  "use strict";

  // Token lives only in this browser's localStorage — the server never
  // sees it except in the X-Axiom-Token header on each request. See
  // core/web_control_server.py for the auth model this pairs with.
  var STORAGE_KEY = "axiom_web_control_token";
  var POLL_MS = 1000;

  var statusEl = document.getElementById("status");
  var tokenInput = document.getElementById("token-input");
  var tokenSaveBtn = document.getElementById("token-save");
  var tokenShowBtn = document.getElementById("token-show");
  var alwaysAimToggle = document.getElementById("always-aim-toggle");
  var makcuToggle = document.getElementById("makcu-toggle");
  var aiStartBtn = document.getElementById("ai-start-btn");
  var aiStopBtn = document.getElementById("ai-stop-btn");
  var aiControlReason = document.getElementById("ai-control-reason");

  function getToken() {
    try {
      return localStorage.getItem(STORAGE_KEY) || "";
    } catch (e) {
      return "";
    }
  }

  function setToken(v) {
    try {
      localStorage.setItem(STORAGE_KEY, v);
    } catch (e) {
      // Private browsing / storage disabled — token just won't persist
      // across reloads; the page still works for the current session.
    }
  }

  tokenInput.value = getToken();

  tokenSaveBtn.addEventListener("click", function () {
    setToken(tokenInput.value.trim());
    poll();
  });

  tokenShowBtn.addEventListener("click", function () {
    tokenInput.type = tokenInput.type === "password" ? "text" : "password";
  });

  function setStatusBadge(connected, label) {
    statusEl.textContent = label || (connected ? "connected" : "disconnected");
    statusEl.className = connected ? "connected" : "disconnected";
  }

  function authHeaders() {
    return {
      "X-Axiom-Token": getToken(),
      "Content-Type": "application/json",
    };
  }

  // Set while we're writing a toggle's .checked from a server response, so
  // that write doesn't re-trigger the toggle's own "change" handler and
  // fire a redundant POST right back at the server. Two independent flags
  // — suppressing one toggle's echo must not accidentally suppress the
  // other's if both happen to update in the same applyStatus() call.
  var suppressToggleEcho = false;
  var suppressMakcuToggleEcho = false;

  function applyStatus(s) {
    document.getElementById("s-running").textContent = s.running ? "RUNNING" : "stopped";
    document.getElementById("s-active").textContent = s.active ? "ON" : "OFF";
    document.getElementById("s-firing").textContent = s.aim_firing ? "FIRING" : "idle";
    document.getElementById("s-model").textContent = s.model || "—";
    document.getElementById("s-backend").textContent = s.inference_backend || "—";
    document.getElementById("s-mouse").textContent = s.mouse_move_method || "—";
    document.getElementById("s-makcu").textContent = s.makcu_connected ? "connected" : "disconnected";
    document.getElementById("s-makcu-port").textContent = s.makcu_com_port || "—";
    document.getElementById("s-capfps").textContent = s.capture_fps != null ? s.capture_fps.toFixed(1) : "—";
    document.getElementById("s-inffps").textContent = s.inference_fps != null ? s.inference_fps.toFixed(1) : "—";

    suppressToggleEcho = true;
    alwaysAimToggle.checked = !!s.always_aim;
    suppressToggleEcho = false;

    // Only reflect server state onto the toggle while it isn't mid-request
    // (disabled) — otherwise a status poll landing between the user's click
    // and the connect/disconnect response finishing could flip it back to
    // the pre-click state for a moment, fighting the in-flight request's own
    // eventual (and authoritative) update.
    if (!makcuToggle.disabled) {
      suppressMakcuToggleEcho = true;
      makcuToggle.checked = !!s.makcu_connected;
      suppressMakcuToggleEcho = false;
    }
  }

  function poll() {
    fetch("/api/status", { headers: authHeaders() })
      .then(function (res) {
        if (res.status === 401) {
          setStatusBadge(false, "bad token");
          return null;
        }
        if (!res.ok) {
          setStatusBadge(false);
          return null;
        }
        return res.json();
      })
      .then(function (data) {
        if (data) {
          setStatusBadge(true);
          applyStatus(data);
        }
      })
      .catch(function () {
        setStatusBadge(false);
      });
  }

  alwaysAimToggle.addEventListener("change", function () {
    if (suppressToggleEcho) return;
    var enabled = alwaysAimToggle.checked;
    fetch("/api/control/always_aim", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ enabled: enabled }),
    })
      .then(function (res) {
        if (!res.ok) {
          // Rejected (bad token, server error) — revert the optimistic UI
          // flip rather than show a control that silently didn't take.
          alwaysAimToggle.checked = !enabled;
        }
      })
      .catch(function () {
        alwaysAimToggle.checked = !enabled;
      });
  });

  makcuToggle.addEventListener("change", function () {
    if (suppressMakcuToggleEcho) return;
    var connecting = makcuToggle.checked;
    var path = connecting ? "/api/control/makcu_connect" : "/api/control/makcu_disconnect";

    makcuToggle.disabled = true;
    fetch(path, { method: "POST", headers: authHeaders() })
      .then(function (res) {
        if (!res.ok) {
          // Bad token / server error — not a normal "connect failed", so
          // there's no JSON body worth reading.
          makcuToggle.checked = !connecting;
          return;
        }
        return res.json();
      })
      .then(function (data) {
        // Unlike always_aim (which can't fail), a MAKCU connect attempt
        // can come back 200 with {"ok": false, "reason": ...} — a bad/
        // unset COM port or a failed handshake, not a request error — so
        // the toggle has to be reverted based on the body, not just
        // res.ok.
        if (data && data.ok === false) {
          makcuToggle.checked = false;
        }
      })
      .catch(function () {
        makcuToggle.checked = !connecting;
      })
      .then(function () {
        makcuToggle.disabled = false;
      });
  });

  function runAiControl(path) {
    aiStartBtn.disabled = true;
    aiStopBtn.disabled = true;
    aiControlReason.textContent = "";
    fetch(path, { method: "POST", headers: authHeaders() })
      .then(function (res) {
        if (!res.ok) {
          aiControlReason.textContent = "request failed";
          return null;
        }
        return res.json();
      })
      .then(function (data) {
        // Neither route has anything to optimistically flip (there's no
        // checkbox here) — on success, the next status poll is the sole
        // source of truth for s-running; on failure, show the reason
        // inline instead.
        if (data && data.ok === false) {
          aiControlReason.textContent = data.reason || "failed";
        }
      })
      .catch(function () {
        aiControlReason.textContent = "request failed";
      })
      .then(function () {
        aiStartBtn.disabled = false;
        aiStopBtn.disabled = false;
      });
  }

  aiStartBtn.addEventListener("click", function () {
    runAiControl("/api/control/ai_start");
  });

  aiStopBtn.addEventListener("click", function () {
    runAiControl("/api/control/ai_stop");
  });

  poll();
  setInterval(poll, POLL_MS);
})();
