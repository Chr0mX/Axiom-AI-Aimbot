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
  var modelSelect = document.getElementById("model-select");
  var backendSelect = document.getElementById("backend-select");
  var modelSwitchBtn = document.getElementById("model-switch-btn");
  var modelSwitchReason = document.getElementById("model-switch-reason");
  var tabButtons = document.querySelectorAll(".tab");
  var panels = document.querySelectorAll(".panel");

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

  // ---------------------------------------------------------------------
  // Tabs — mirrors the real Qt app's NavigationInterface order (see
  // CLAUDE.md's GUI section). Purely a visibility toggle: poll()/
  // setInterval keep running regardless of which tab is active, since the
  // sidebar (and its toggles' echo-suppression flags below) is always
  // visible no matter which panel is showing.
  // ---------------------------------------------------------------------
  function activateTab(name) {
    tabButtons.forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === name);
    });
    panels.forEach(function (panel) {
      panel.classList.toggle("active", panel.id === "panel-" + name);
    });
    // Re-fetch this tab's settings every time it's activated (not just
    // once) so a change made from the Qt GUI while this tab was in the
    // background is picked up on return.
    if (name === "model" || name === "capture" || name === "inference") {
      loadTabSettings(name);
    }
    if (name === "model") {
      ensureModelPanelExtras();
    }
  }

  tabButtons.forEach(function (btn) {
    btn.addEventListener("click", function () {
      activateTab(btn.getAttribute("data-tab"));
    });
  });

  // Set while we're writing a toggle's .checked from a server response, so
  // that write doesn't re-trigger the toggle's own "change" handler and
  // fire a redundant POST right back at the server. Two independent flags
  // — suppressing one toggle's echo must not accidentally suppress the
  // other's if both happen to update in the same applyStatus() call.
  var suppressToggleEcho = false;
  var suppressMakcuToggleEcho = false;

  // The model select is populated once from GET /api/models (loadModelList,
  // fetched once on page load — not every poll tick) and the "currently
  // active" entry is applied once from the first status poll, same one-shot
  // semantics the old text-input prefill had. The two requests can land in
  // either order, so pendingModelSelection holds the status poll's answer
  // if it arrives before the model list has options to select among.
  var modelFormPrefilled = false;
  var modelListLoaded = false;
  var pendingModelSelection = null;
  var allModelNames = []; // master list — modelSearchInput filters a view of this

  function applyModelSelection(name) {
    if (!name) return;
    if (!modelListLoaded) {
      pendingModelSelection = name;
      return;
    }
    var hasOption = Array.prototype.some.call(modelSelect.options, function (opt) {
      return opt.value === name;
    });
    if (hasOption) {
      modelSelect.value = name;
      refreshModelInfoAndNotes();
    }
    // else: the active model isn't in Model/ anymore (renamed/deleted since
    // it was loaded) — leave the select on its default rather than adding
    // a phantom option for a file that may no longer exist.
  }

  function loadModelList() {
    fetch("/api/models", { headers: authHeaders() })
      .then(function (res) {
        return res.ok ? res.json() : null;
      })
      .then(function (data) {
        if (!data || !data.models) return;
        allModelNames = data.models;
        modelSelect.innerHTML = "";
        data.models.forEach(function (name) {
          var opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          modelSelect.appendChild(opt);
        });
        modelListLoaded = true;
        modelSwitchBtn.disabled = false;
        if (pendingModelSelection) {
          applyModelSelection(pendingModelSelection);
          pendingModelSelection = null;
        }
      })
      .catch(function () {});
  }

  // Live-filters modelSelect to entries matching the search box (case-
  // insensitive substring), mirroring model_page.py's modelSearchEdit —
  // but never hides the currently active model, so narrowing the search
  // can't make the select appear to show the wrong "currently active"
  // model.
  var modelSearchInput = document.getElementById("model-search-input");
  modelSearchInput.addEventListener("input", function () {
    var query = modelSearchInput.value.trim().toLowerCase();
    var current = modelSelect.value;
    var matchedCurrent = !current;
    modelSelect.innerHTML = "";
    allModelNames.forEach(function (name) {
      if (!query || name.toLowerCase().indexOf(query) !== -1) {
        var opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        modelSelect.appendChild(opt);
        if (name === current) matchedCurrent = true;
      }
    });
    if (!matchedCurrent && current) {
      var opt2 = document.createElement("option");
      opt2.value = current;
      opt2.textContent = current;
      modelSelect.appendChild(opt2);
    }
    modelSelect.value = current;
  });

  // ---------------------------------------------------------------------
  // Model Info / Model Notes — mirrors model_page.py's live inspector and
  // per-model notes card. Refreshed whenever the model selection settles
  // (user pick, or the status-driven one-shot prefill above).
  // ---------------------------------------------------------------------
  function refreshModelInfoAndNotes() {
    var name = modelSelect.value;
    var infoEl = document.getElementById("model-info-text");
    var notesEl = document.getElementById("model-notes-text");
    if (!name) {
      infoEl.textContent = "—";
      notesEl.value = "";
      return;
    }
    fetch("/api/model_info?model=" + encodeURIComponent(name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) { infoEl.textContent = "—"; return; }
        if (data.ok === false) {
          infoEl.textContent = "Unavailable (" + (data.reason || "error") + ")";
          return;
        }
        infoEl.textContent = data.text || "—";
      })
      .catch(function () { infoEl.textContent = "—"; });

    fetch("/api/model_notes?model=" + encodeURIComponent(name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        notesEl.value = data && data.text ? data.text : "";
      })
      .catch(function () {});
  }

  modelSelect.addEventListener("change", refreshModelInfoAndNotes);

  var modelNotesEditBtn = document.getElementById("model-notes-edit-btn");
  var modelNotesTextarea = document.getElementById("model-notes-text");
  modelNotesEditBtn.addEventListener("click", function () {
    if (modelNotesTextarea.readOnly) {
      modelNotesTextarea.readOnly = false;
      modelNotesTextarea.focus();
      modelNotesEditBtn.textContent = "Save";
    } else {
      modelNotesTextarea.readOnly = true;
      modelNotesEditBtn.textContent = "Edit";
      var name = modelSelect.value;
      if (!name) return;
      fetch("/api/model_notes", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ model: name, text: modelNotesTextarea.value }),
      }).catch(function () {});
    }
  });

  var openModelFolderBtn = document.getElementById("open-model-folder-btn");
  openModelFolderBtn.addEventListener("click", function () {
    openModelFolderBtn.disabled = true;
    fetch("/api/control/open_model_folder", { method: "POST", headers: authHeaders() })
      .catch(function () {})
      .then(function () { openModelFolderBtn.disabled = false; });
  });

  // ---------------------------------------------------------------------
  // Generic tab settings — one pair of routes (GET/POST /api/settings/
  // {tab}) covers every plain Config field on the Model/Capture/Inference
  // panels. Elements opt in via data-key/data-type; data-custom="1"
  // excludes an element from the generic *write* path only (its value is
  // still read/applied generically) — used by fields whose write is a
  // coupled multi-field action (e.g. picking a game profile also writes
  // hud_roi_coords) instead of a single setattr.
  // ---------------------------------------------------------------------
  var lastSettings = { model: {}, capture: {}, inference: {} };

  function fieldElsFor(tab) {
    return document.querySelectorAll('#panel-' + tab + ' [data-key]');
  }

  function applyGenericValue(el, type, value) {
    if (value === null || value === undefined) return;
    if (type === "bool") {
      el.checked = !!value;
    } else if (type === "dshow-toggle") {
      el.checked = value === "v2";
    } else {
      el.value = value;
    }
  }

  function readGenericValue(el, type) {
    if (type === "bool") return el.checked;
    if (type === "dshow-toggle") return el.checked ? "v2" : "v1";
    if (type === "number") return parseFloat(el.value);
    return el.value;
  }

  function loadTabSettings(tab) {
    fetch("/api/settings/" + tab, { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) return;
        lastSettings[tab] = data;
        if (tab === "capture") applyCaptureExtras(data);
        Array.prototype.forEach.call(fieldElsFor(tab), function (el) {
          if (el === document.activeElement) return; // don't clobber an in-progress edit
          var key = el.getAttribute("data-key");
          var type = el.getAttribute("data-type");
          if (!(key in data)) return;
          applyGenericValue(el, type, data[key]);
        });
        if (tab === "capture") updateCaptureVisibility(data.screenshot_method);
        if (tab === "inference") updateFovReduceVisibility(!!data.fov_reduce_on_target_enabled);
      })
      .catch(function () {});
  }

  function pushTabSetting(tab, key, value) {
    var body = {};
    body[key] = value;
    fetch("/api/settings/" + tab, {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          // Rejected — resync this tab's real values instead of leaving
          // the UI showing an edit that never actually applied.
          loadTabSettings(tab);
        }
      })
      .catch(function () { loadTabSettings(tab); });
  }

  function wireGenericFields(tab) {
    Array.prototype.forEach.call(fieldElsFor(tab), function (el) {
      if (el.hasAttribute("data-custom")) return;
      el.addEventListener("change", function () {
        var key = el.getAttribute("data-key");
        var type = el.getAttribute("data-type");
        var value = readGenericValue(el, type);
        pushTabSetting(tab, key, value);
        if (tab === "capture" && key === "screenshot_method") updateCaptureVisibility(value);
        if (tab === "capture" && (key === "uvc_capture_method" || key === "uvc_dshow_backend" || key === "uvc_ffmpeg_enabled")) {
          updateUvcSubVisibility();
        }
        if (tab === "inference" && key === "fov_reduce_on_target_enabled") updateFovReduceVisibility(el.checked);
      });
    });
  }

  function updateFovReduceVisibility(enabled) {
    document.getElementById("inf-fov-min-size-card").classList.toggle("hidden", !enabled);
    document.getElementById("inf-fov-min-duration-card").classList.toggle("hidden", !enabled);
  }

  // ---------------------------------------------------------------------
  // Capture panel — conditional UVC/NDI/UDP groups + device/source probing
  // ---------------------------------------------------------------------
  function ensureOption(select, value, label) {
    if (value === undefined || value === null || value === "") return;
    var text = String(value);
    var has = Array.prototype.some.call(select.options, function (opt) { return opt.value === text; });
    if (!has) {
      var opt = document.createElement("option");
      opt.value = text;
      opt.textContent = label != null ? label : text;
      select.appendChild(opt);
    }
  }

  function applyCaptureExtras(data) {
    document.getElementById("cap-udp-system-ip").textContent = "Stream to: " + (data.system_ip || "—");

    var bindSelect = document.getElementById("cap-udp-bind-ip-select");
    var bindOptions = data.bind_ip_options || ["0.0.0.0"];
    bindSelect.innerHTML = "";
    bindOptions.forEach(function (ip) {
      var opt = document.createElement("option");
      opt.value = ip;
      opt.textContent = ip;
      bindSelect.appendChild(opt);
    });
    if (data.udp_bind_ip) {
      ensureOption(bindSelect, data.udp_bind_ip);
      bindSelect.value = data.udp_bind_ip;
    }

    var deviceSelect = document.getElementById("cap-uvc-device-select");
    ensureOption(deviceSelect, data.uvc_device_index, "Device " + data.uvc_device_index);
    deviceSelect.value = String(data.uvc_device_index);

    var resSelect = document.getElementById("cap-uvc-resolution-select");
    if (data.uvc_width && data.uvc_height) {
      var resText = data.uvc_width + "x" + data.uvc_height;
      ensureOption(resSelect, resText);
      resSelect.value = resText;
    }

    var fpsSelect = document.getElementById("cap-uvc-fps-select");
    ensureOption(fpsSelect, data.uvc_fps);
    fpsSelect.value = String(data.uvc_fps);

    var ndiSelect = document.getElementById("cap-ndi-source-select");
    if (data.ndi_source_name) {
      ensureOption(ndiSelect, data.ndi_source_name);
      ndiSelect.value = data.ndi_source_name;
    }

    var hwInfo = document.getElementById("cap-uvc-hw-info");
    if (data.uvc_actual_width && data.uvc_actual_height) {
      var fpsStr = data.uvc_actual_fps ? data.uvc_actual_fps.toFixed(1) : "?";
      hwInfo.textContent = data.uvc_actual_width + " × " + data.uvc_actual_height + " @ " + fpsStr + " fps";
    } else {
      hwInfo.textContent = "—  (device not available)";
    }

    var ndiHwInfo = document.getElementById("cap-ndi-hw-info");
    if (data.ndi_width && data.ndi_height) {
      var ndiFpsStr = data.ndi_source_nominal_fps ? data.ndi_source_nominal_fps.toFixed(1) : "?";
      ndiHwInfo.textContent = data.ndi_width + " × " + data.ndi_height + " @ " + ndiFpsStr + " fps";
    } else {
      ndiHwInfo.textContent = "—  (connect source to see info)";
    }
  }

  function updateCaptureVisibility(method) {
    var isUvc = method === "uvc", isNdi = method === "ndi", isUdp = method === "udp";
    ["cap-uvc-group", "cap-uvc-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", !isUvc);
    });
    ["cap-ndi-group", "cap-ndi-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", !isNdi);
    });
    ["cap-udp-group", "cap-udp-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", !isUdp);
    });
    if (isUvc) updateUvcSubVisibility();
  }

  function updateUvcSubVisibility() {
    var methodEl = document.getElementById("cap-uvc_capture_method");
    var isDshow = !!(methodEl && methodEl.value === "dshow");
    var v2El = document.getElementById("cap-uvc_dshow_backend");
    var isV1 = !(v2El && v2El.checked);
    var ffEl = document.getElementById("cap-uvc_ffmpeg_enabled");
    var ffOn = !!(ffEl && ffEl.checked);
    document.getElementById("cap-uvc-dshow-v2-card").classList.toggle("hidden", !isDshow);
    document.getElementById("cap-uvc-ffmpeg-enabled-card").classList.toggle("hidden", !(isDshow && isV1));
    document.getElementById("cap-uvc-ffmpeg-path-card").classList.toggle("hidden", !(isDshow && isV1 && ffOn));
  }

  document.getElementById("cap-uvc-resolution-select").addEventListener("change", function (e) {
    // Coupled write (uvc_width + uvc_height) — two independent single-key
    // POSTs rather than one combined body, keeping every pushTabSetting()
    // call symmetric with the generic single-field path.
    var parts = e.target.value.split("x");
    if (parts.length !== 2) return;
    var w = parseInt(parts[0], 10), h = parseInt(parts[1], 10);
    if (!w || !h) return;
    pushTabSetting("capture", "uvc_width", w);
    pushTabSetting("capture", "uvc_height", h);
  });

  document.getElementById("cap-ndi-source-select").addEventListener("change", function (e) {
    var name = e.target.value;
    if (!name) return;
    pushTabSetting("capture", "ndi_source_name", name);
    pushTabSetting("capture", "ndi_force_reconnect", true);
  });

  document.getElementById("cap-udp-restart-btn").addEventListener("click", function () {
    pushTabSetting("capture", "udp_force_restart", true);
  });

  document.getElementById("cap-uvc-refresh-btn").addEventListener("click", function () {
    var btn = document.getElementById("cap-uvc-refresh-btn");
    var cap = lastSettings.capture || {};
    var device = cap.uvc_device_index || 0;
    var method = cap.uvc_capture_method || "msmf";
    var width = cap.uvc_width || 1920;
    var height = cap.uvc_height || 1080;
    btn.disabled = true;
    var url = "/api/uvc_probe?device=" + encodeURIComponent(device) +
      "&method=" + encodeURIComponent(method) +
      "&width=" + encodeURIComponent(width) +
      "&height=" + encodeURIComponent(height);
    fetch(url, { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) return;
        var deviceSelect = document.getElementById("cap-uvc-device-select");
        var currentDevice = deviceSelect.value;
        deviceSelect.innerHTML = "";
        (data.device_names || []).forEach(function (name, i) {
          var opt = document.createElement("option");
          opt.value = String(i);
          opt.textContent = name;
          deviceSelect.appendChild(opt);
        });
        ensureOption(deviceSelect, currentDevice, "Device " + currentDevice);
        deviceSelect.value = currentDevice;

        var resSelect = document.getElementById("cap-uvc-resolution-select");
        var currentRes = resSelect.value;
        resSelect.innerHTML = "";
        (data.resolutions || []).forEach(function (pair) {
          var text = pair[0] + "x" + pair[1];
          var opt = document.createElement("option");
          opt.value = text;
          opt.textContent = text;
          resSelect.appendChild(opt);
        });
        ensureOption(resSelect, currentRes);
        resSelect.value = currentRes;

        var fpsSelect = document.getElementById("cap-uvc-fps-select");
        var currentFps = fpsSelect.value;
        fpsSelect.innerHTML = "";
        (data.fps_list || []).forEach(function (fps) {
          var opt = document.createElement("option");
          opt.value = String(fps);
          opt.textContent = String(fps);
          fpsSelect.appendChild(opt);
        });
        ensureOption(fpsSelect, currentFps);
        fpsSelect.value = currentFps;
      })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  });

  document.getElementById("cap-ndi-refresh-btn").addEventListener("click", function () {
    var btn = document.getElementById("cap-ndi-refresh-btn");
    btn.disabled = true;
    fetch("/api/ndi_sources", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) return;
        var select = document.getElementById("cap-ndi-source-select");
        var current = select.value;
        select.innerHTML = "";
        (data.sources || []).forEach(function (src) {
          var opt = document.createElement("option");
          opt.value = src.name || "";
          opt.textContent = src.label || src.name || "";
          select.appendChild(opt);
        });
        ensureOption(select, current);
        if (current) select.value = current;
      })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  });

  // ---------------------------------------------------------------------
  // Model HUD Settings — Game Profile (coupled write: also sets
  // hud_roi_coords from game.json) + HUD Model list, both fetched once.
  // ---------------------------------------------------------------------
  var gameProfilesLoaded = false;
  var hudModelsLoaded = false;

  document.getElementById("hud-game-select").addEventListener("change", function (e) {
    var opt = e.target.options[e.target.selectedIndex];
    var roi = opt ? (opt.getAttribute("data-roi") || "") : "";
    pushTabSetting("model", "hud_game", e.target.value);
    pushTabSetting("model", "hud_roi_coords", roi);
  });

  function ensureModelPanelExtras() {
    if (!gameProfilesLoaded) {
      gameProfilesLoaded = true;
      fetch("/api/game_profiles", { headers: authHeaders() })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (!data) return;
          var select = document.getElementById("hud-game-select");
          var games = data.games || {};
          select.innerHTML = "";
          Object.keys(games).forEach(function (name) {
            var opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            opt.setAttribute("data-roi", games[name] || "");
            select.appendChild(opt);
          });
          if (lastSettings.model.hud_game) select.value = lastSettings.model.hud_game;
        })
        .catch(function () {});
    }
    if (!hudModelsLoaded) {
      hudModelsLoaded = true;
      fetch("/api/hud_models", { headers: authHeaders() })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (!data || !data.models) return;
          var select = document.getElementById("hud-model-select");
          select.innerHTML = "";
          data.models.forEach(function (name) {
            var opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
          });
          if (lastSettings.model.hud_model_path) select.value = lastSettings.model.hud_model_path;
        })
        .catch(function () {});
    }
  }

  // The backend select offers only 4 options with no separate "CUDA" entry
  // (mirrors model_page.py's own reverse-map fold, which folds "cuda" onto
  // the "TensorRT" display item) — a "cuda" backend from the server is
  // shown as "TensorRT" here, but never written back as "cuda" in a POST
  // body, matching model_page.py's write-side map which also never writes
  // "cuda".
  function displayBackend(name) {
    return name === "cuda" ? "tensorrt" : name;
  }

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

    if (!modelFormPrefilled && s.model) {
      applyModelSelection(s.model);
      modelFormPrefilled = true;
    }
    if (s.inference_backend) {
      var backendDisplay = displayBackend(s.inference_backend);
      var hasOption = Array.prototype.some.call(backendSelect.options, function (opt) {
        return opt.value === backendDisplay;
      });
      if (hasOption) backendSelect.value = backendDisplay;
      // else: a valid backend the route accepts but the GUI (and this
      // select, mirroring it) doesn't offer as a pickable option — leave
      // the select on whatever the operator last chose rather than
      // silently landing on an unrelated option.
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

  modelSwitchBtn.addEventListener("click", function () {
    // modelSelect's options are bare basenames (see loadModelList()) —
    // resolve_model_path() joins a relative path directly against
    // project_root, not project_root/Model, so the "Model/" prefix has to
    // be added client-side.
    var modelPath = "Model/" + modelSelect.value;
    modelSwitchBtn.disabled = true;
    modelSwitchReason.textContent = "";
    fetch("/api/control/model", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ model_path: modelPath, inference_backend: backendSelect.value }),
    })
      .then(function (res) {
        if (!res.ok) {
          modelSwitchReason.textContent = "request failed";
          return null;
        }
        return res.json();
      })
      .then(function (data) {
        if (!data) return;
        if (data.ok === false) {
          modelSwitchReason.textContent = data.reason || "failed";
        } else {
          // Never optimistically write s-model/s-backend here — the next
          // status poll is the sole source of truth for those, same as
          // every other field this client doesn't own a checkbox for.
          modelSwitchReason.textContent = data.applied_live
            ? "applied — live"
            : "applied — takes effect on next AI start";
        }
      })
      .catch(function () {
        modelSwitchReason.textContent = "request failed";
      })
      .then(function () {
        modelSwitchBtn.disabled = false;
      });
  });

  // Model select starts disabled-for-submit until the real list arrives —
  // submitting "Model/" (an empty selection) before then would always
  // resolve to "not_found" rather than doing anything useful.
  modelSwitchBtn.disabled = true;

  wireGenericFields("model");
  wireGenericFields("capture");
  wireGenericFields("inference");

  loadModelList();
  // Model is the default-active panel (no click to trigger activateTab's
  // own load), so its tab settings + HUD extras are fetched explicitly here.
  loadTabSettings("model");
  ensureModelPanelExtras();
  poll();
  setInterval(poll, POLL_MS);
})();
