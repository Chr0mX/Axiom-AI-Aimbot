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

  // Set while we're writing alwaysAimToggle.checked from a server response,
  // so that write doesn't re-trigger the toggle's own "change" handler and
  // fire a redundant POST right back at the server.
  var suppressToggleEcho = false;

  function applyStatus(s) {
    document.getElementById("s-active").textContent = s.active ? "ON" : "OFF";
    document.getElementById("s-firing").textContent = s.aim_firing ? "FIRING" : "idle";
    document.getElementById("s-model").textContent = s.model || "—";
    document.getElementById("s-backend").textContent = s.inference_backend || "—";
    document.getElementById("s-mouse").textContent = s.mouse_move_method || "—";
    document.getElementById("s-makcu").textContent = s.makcu_connected ? "connected" : "disconnected";
    document.getElementById("s-capfps").textContent = s.capture_fps != null ? s.capture_fps.toFixed(1) : "—";
    document.getElementById("s-inffps").textContent = s.inference_fps != null ? s.inference_fps.toFixed(1) : "—";

    suppressToggleEcho = true;
    alwaysAimToggle.checked = !!s.always_aim;
    suppressToggleEcho = false;
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

  poll();
  setInterval(poll, POLL_MS);
})();
