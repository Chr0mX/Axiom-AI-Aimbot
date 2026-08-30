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
  // Every tab backed by the generic get_tab_settings()/apply_tab_settings()
  // mechanism — used both to decide whether activateTab() should fetch it
  // and, further down, to keep whichever one is currently on-screen synced
  // on every poll() tick (not just on activation), so a change made from
  // the Qt GUI while this page is sitting open surfaces within ~1s with no
  // reload. See loadTabSettings()'s "don't clobber an in-progress edit"
  // guard for why this can't just blindly overwrite fields the user is
  // mid-edit on.
  var GENERIC_SCHEMA_TABS = ["model", "capture", "inference", "aim", "keys", "visuals", "trigger", "convert"];

  // The tab currently on-screen — updated by activateTab(), read by
  // poll()'s continuous-resync tick below.
  var currentTab = "model";

  function activateTab(name) {
    currentTab = name;
    tabButtons.forEach(function (btn) {
      btn.classList.toggle("active", btn.getAttribute("data-tab") === name);
    });
    panels.forEach(function (panel) {
      panel.classList.toggle("active", panel.id === "panel-" + name);
    });
    // Re-fetch this tab's settings every time it's activated (not just
    // once) so a change made from the Qt GUI while this tab was in the
    // background is picked up on return.
    if (GENERIC_SCHEMA_TABS.indexOf(name) !== -1) {
      loadTabSettings(name);
    }
    if (name === "model") {
      ensureModelPanelExtras();
    }
    if (name === "keys") {
      ensureVkOptionsLoaded();
    }
    if (name === "configs") {
      refreshConfigsList();
    }
    if (name === "convert") {
      ensureConvertPanelExtras();
    }
    // Stops the preview stream the instant the operator navigates away
    // from Capture (updatePreviewStream() checks currentTab itself) —
    // loadTabSettings("capture")'s own response re-evaluates this again
    // once it lands, in case screenshot_method also changed meanwhile.
    updatePreviewStream();
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
    // /api/model_info resolves via resolve_model_path(), which joins a
    // relative path directly against project_root, not project_root/Model
    // — same "Model/" prefix the Switch button already adds, needed here
    // too since modelSelect's options are bare basenames.
    fetch("/api/model_info?model=" + encodeURIComponent("Model/" + name), { headers: authHeaders() })
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

  // Almost every numeric field across the Qt app pairs a Slider with its
  // SpinBox/DoubleSpinBox (SliderSpinCard/SliderDoubleSpinCard/
  // SliderLabelCard — see CLAUDE.md's slider_spin_card.py note): FOV size,
  // PID gains, humanization params, target-area thresholds, MAKCU
  // disengage delay, auto-fire delay/interval, and so on. This gives every
  // input[type=number][data-key] in this client the same treatment, driven
  // entirely by the number input's own min/max/step attributes (already
  // present on every one of them for client-side validation) rather than
  // hand-writing a paired <input type="range"> into ~80 individual
  // field-cards. The two Web ESP port fields are the one deliberate
  // exception — they're plain SpinBoxes with no Slider companion in Qt
  // (visuals_page.py never pairs a port number with a slider).
  var SLIDER_EXCLUDED_KEYS = ["web_esp_http_port", "web_esp_ws_port"];

  // Fixed dimensions from styles.css's input[type="range"] / ::-webkit-
  // slider-thumb rules — kept in sync with those, not read from the DOM
  // (see the thumb-inset comment below for why).
  var RANGE_TRACK_PX = 160;
  var RANGE_THUMB_PX = 18;

  // Drives the filled-progress look in styles.css (the --range-progress
  // custom property a linear-gradient background reads) — called on every
  // value change from any direction (drag, typing in the paired number
  // box, or a server-driven refresh) so the fill never goes stale.
  //
  // A naive fraction*100% split (the first version of this) visibly drifts
  // from where the thumb actually sits — a real, reported bug ("slider not
  // properly aligned to the slider [thumb]"): the browser insets the
  // thumb's travel by half its own width at each end (its center can never
  // reach x=0 or x=trackWidth, only x=thumbWidth/2 and x=trackWidth-
  // thumbWidth/2), but a flat value-fraction split assumes the fill
  // boundary reaches all the way to both edges. The mismatch is worst near
  // either end (up to ~half the thumb's own width, ~9px here) and zero
  // only exactly at the midpoint. Computing the thumb's true center
  // position first and expressing THAT as a percentage of the track
  // eliminates it. offsetWidth isn't used here on purpose — most of these
  // ranges live inside a `.panel` that's still `display:none` (not yet
  // activated) the first time enhanceNumberInputsWithSliders() runs at
  // page load, where offsetWidth would read 0 and this whole calculation
  // would divide by zero; the track/thumb widths are fixed in CSS anyway,
  // so hardcoding them (kept in sync via the comment above) avoids that
  // trap entirely.
  function updateRangeFill(range) {
    var min = parseFloat(range.min);
    var max = parseFloat(range.max);
    var val = parseFloat(range.value);
    var fraction = max > min ? (val - min) / (max - min) : 0;
    fraction = Math.max(0, Math.min(1, fraction));
    var thumbCenterPx = RANGE_THUMB_PX / 2 + fraction * (RANGE_TRACK_PX - RANGE_THUMB_PX);
    var pct = (thumbCenterPx / RANGE_TRACK_PX) * 100;
    range.style.setProperty("--range-progress", pct + "%");
  }

  function enhanceNumberInputsWithSliders() {
    var inputs = document.querySelectorAll('input[type="number"][data-key]');
    Array.prototype.forEach.call(inputs, function (numberInput) {
      var key = numberInput.getAttribute("data-key");
      if (SLIDER_EXCLUDED_KEYS.indexOf(key) !== -1) return;
      var min = numberInput.getAttribute("min");
      var max = numberInput.getAttribute("max");
      if (min === null || max === null || min === "" || max === "") return;

      var range = document.createElement("input");
      range.type = "range";
      range.min = min;
      range.max = max;
      range.step = numberInput.getAttribute("step") || "1";
      range.value = numberInput.value || min;
      range.className = "field-slider";
      updateRangeFill(range);

      // Inserted as a plain sibling inside the number input's own
      // .field-control row (slider, then number, then any unit suffix) —
      // it carries no data-key of its own, so it never enters
      // fieldElsFor()'s generic read/write loop; it's purely a second View
      // over the exact same number input, wired below to mirror it in
      // both directions and to re-dispatch a real "change" event so
      // wireGenericFields()'s existing listener (already attached to the
      // number input) fires exactly as if the number input itself had
      // been edited. No new server-communication path needed at all.
      numberInput.parentNode.insertBefore(range, numberInput);
      numberInput._pairedRange = range;
      // Lets styles.css narrow the number box once a slider sits next to
      // it — see .field-control input[type="number"].has-slider.
      numberInput.classList.add("has-slider");

      range.addEventListener("input", function () {
        numberInput.value = range.value;
        updateRangeFill(range);
      });
      range.addEventListener("change", function () {
        numberInput.value = range.value;
        numberInput.dispatchEvent(new Event("change", { bubbles: true }));
      });
      numberInput.addEventListener("input", function () {
        range.value = numberInput.value;
        updateRangeFill(range);
      });
    });
  }

  function applyGenericValue(el, type, value) {
    if (value === null || value === undefined) return;
    if (type === "bool") {
      el.checked = !!value;
    } else if (type === "dshow-toggle") {
      el.checked = value === "v2";
    } else {
      el.value = value;
      // Keep a paired range slider (see enhanceNumberInputsWithSliders())
      // in sync too — applyGenericValue() sets .value directly rather than
      // dispatching an "input" event, so the slider's own mirroring
      // listener never fires on its own for a server-driven refresh.
      if (el._pairedRange) {
        el._pairedRange.value = value;
        updateRangeFill(el._pairedRange);
      }
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
          // ...nor while the user is mid-drag on its paired range slider —
          // dragging focuses the range element, not this number input, so
          // the check above alone wouldn't catch it.
          if (el._pairedRange && el._pairedRange === document.activeElement) return;
          var key = el.getAttribute("data-key");
          var type = el.getAttribute("data-type");
          if (!(key in data)) return;
          applyGenericValue(el, type, data[key]);
        });
        if (tab === "capture") updateCaptureVisibility(data.screenshot_method);
        if (tab === "inference") updateFovReduceVisibility(!!data.fov_reduce_on_target_enabled);
        if (tab === "aim") updateHumanizationVisibility();
        if (tab === "keys") updateKeysVisibility(data);
        if (tab === "visuals") applyVisualsExtras(data);
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
        if (tab === "aim" && HUMANIZATION_TOGGLE_KEYS.indexOf(key) !== -1) updateHumanizationVisibility();
        if (tab === "visuals" && (key === "web_esp_http_port" || key === "web_esp_ws_port")) {
          scheduleWebEspRestart();
        }
        // One-directional coupling, mirroring trigger_page.py's
        // _onAlwaysAutoFireChanged(): turning Always Auto-Fire ON also
        // disables Keep Detecting While Idle — turning it back off does
        // NOT re-enable idle detect (Qt doesn't either). idle_detect_enabled
        // lives on the "inference" tab's own schema, so this is a second,
        // independent POST to a different tab's route, not a same-tab
        // coupled body — same "client already has both values in hand"
        // precedent as every other coupled write in this file.
        if (tab === "trigger" && key === "always_auto_fire" && value === true) {
          pushTabSetting("inference", "idle_detect_enabled", false);
        }
      });
    });
  }

  function updateFovReduceVisibility(enabled) {
    document.getElementById("inf-fov-min-size-card").classList.toggle("hidden", !enabled);
    document.getElementById("inf-fov-min-duration-card").classList.toggle("hidden", !enabled);
  }

  // ---------------------------------------------------------------------
  // Aim panel — Humanization sub-field visibility, PID Unsafe Mode's
  // clamp-on-disable, Move Method's mouse_click_method coupling, and the
  // Humanization Reset-to-Defaults action.
  // ---------------------------------------------------------------------
  var HUMANIZATION_TOGGLE_KEYS = [
    "humanization.enabled",
    "humanization.micro_jitter_enabled",
    "humanization.motion_variation_enabled",
    "humanization.speed_shaping_enabled",
    "humanization.micro_stutter_enabled",
    "humanization.reaction_variability_enabled",
  ];

  function updateHumanizationVisibility() {
    var master = document.getElementById("aim-humanization-enabled").checked;
    function gate(toggleId, groupId) {
      var toggle = document.getElementById(toggleId);
      var enabled = master && !!(toggle && toggle.checked);
      document.getElementById(groupId).classList.toggle("hidden", !enabled);
    }
    gate("aim-humanization-micro-jitter-enabled", "aim-humanization-jitter-subgroup");
    gate("aim-humanization-motion-variation-enabled", "aim-humanization-motion-variation-subgroup");
    gate("aim-humanization-speed-shaping-enabled", "aim-humanization-speed-shaping-subgroup");
    gate("aim-humanization-micro-stutter-enabled", "aim-humanization-stutter-subgroup");
    gate("aim-humanization-reaction-variability-enabled", "aim-humanization-reaction-subgroup");
  }

  document.getElementById("aim-humanization-reset-btn").addEventListener("click", function () {
    var btn = document.getElementById("aim-humanization-reset-btn");
    btn.disabled = true;
    fetch("/api/control/humanization_reset", { method: "POST", headers: authHeaders() })
      .catch(function () {})
      .then(function () {
        loadTabSettings("aim"); // re-pull the fresh HumanizationConfig() defaults
        btn.disabled = false;
      });
  });

  // pid_unsafe_mode is data-custom — turning it OFF must also clamp
  // pid_kp_x/y down to <=0.5, mirroring aim_page.py's _onPidUnsafeChanged().
  // Done as one coupled POST (all three keys the client already has in
  // hand from the currently-loaded number inputs), same precedent as
  // ndi_source_name/ndi_force_reconnect.
  document.getElementById("aim-pid_unsafe_mode").addEventListener("change", function (e) {
    var enabled = e.target.checked;
    var body = { pid_unsafe_mode: enabled };
    if (!enabled) {
      var kpX = document.getElementById("aim-pid_kp_x");
      var kpY = document.getElementById("aim-pid_kp_y");
      var xVal = parseFloat(kpX.value);
      var yVal = parseFloat(kpY.value);
      if (!isNaN(xVal) && xVal > 0.5) { body.pid_kp_x = 0.5; kpX.value = "0.5"; }
      if (!isNaN(yVal) && yVal > 0.5) { body.pid_kp_y = 0.5; kpY.value = "0.5"; }
    }
    fetch("/api/settings/aim", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) loadTabSettings("aim");
      })
      .catch(function () { loadTabSettings("aim"); });
  });

  // mouse_move_method is data-custom — switching to "makcu" must also set
  // mouse_click_method="makcu", mirroring aim_page.py's
  // _onMouseMoveChanged(). ensure_ddxoft_ready() and the cross-page
  // keysInterface visibility refresh the GUI does on this same change have
  // no remote equivalent — the Keys & HW panel already re-fetches its own
  // visibility state every time it's activated, so it self-heals without
  // needing a push from here.
  document.getElementById("aim-mouse_move_method").addEventListener("change", function (e) {
    var method = e.target.value;
    var body = { mouse_move_method: method };
    if (method === "makcu") body.mouse_click_method = "makcu";
    fetch("/api/settings/aim", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify(body),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) loadTabSettings("aim");
      })
      .catch(function () { loadTabSettings("aim"); });
  });

  // ---------------------------------------------------------------------
  // Keys & HW panel — MAKCU-mode visibility, hotkey VK-select population,
  // MAKCU COM port enumeration.
  // ---------------------------------------------------------------------
  function updateKeysVisibility(data) {
    var isMakcu = data.mouse_move_method === "makcu";
    ["keys-aim-group", "keys-aim-group-title", "keys-fire-group", "keys-fire-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", isMakcu);
    });
    ["keys-makcu-conn-group", "keys-makcu-conn-group-title", "keys-makcu-keys-group", "keys-makcu-keys-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", !isMakcu);
    });
    document.getElementById("keys-makcu-inference-card").classList.toggle("hidden", !!data.keep_detecting);
    // always_aim-gated cards are also kept live-synced from every status
    // poll (see applyStatus()) regardless of which tab is active — set
    // here too so a fresh tab activation doesn't show a stale state for
    // the instant before the next poll tick lands.
    var hideForAlwaysAim = !!data.always_aim;
    document.getElementById("keys-makcu-trigger-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-aim-mode-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-disengage-card").classList.toggle("hidden", hideForAlwaysAim);
  }

  var vkOptionsLoaded = false;

  function ensureVkOptionsLoaded() {
    if (vkOptionsLoaded) return;
    vkOptionsLoaded = true;
    fetch("/api/vk_options", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || !data.options) return;
        var selects = document.querySelectorAll(".vk-select");
        selects.forEach(function (select) {
          select.innerHTML = "";
          data.options.forEach(function (opt) {
            var el = document.createElement("option");
            el.value = String(opt.code);
            el.textContent = opt.label;
            select.appendChild(el);
          });
        });
        // Options didn't exist yet the first time loadTabSettings("keys")
        // ran (at tab-activation time) — re-apply now that they do.
        loadTabSettings("keys");
      })
      .catch(function () {});
  }

  document.getElementById("keys-makcu-port-refresh-btn").addEventListener("click", function () {
    var btn = document.getElementById("keys-makcu-port-refresh-btn");
    btn.disabled = true;
    fetch("/api/serial_ports", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) return;
        var select = document.getElementById("keys-makcu_com_port");
        var current = select.value;
        select.innerHTML = "";
        (data.ports || []).forEach(function (port) {
          var opt = document.createElement("option");
          opt.value = port;
          opt.textContent = port;
          select.appendChild(opt);
        });
        if (current) {
          var has = Array.prototype.some.call(select.options, function (o) { return o.value === current; });
          if (!has) {
            var opt2 = document.createElement("option");
            opt2.value = current;
            opt2.textContent = current;
            select.appendChild(opt2);
          }
          select.value = current;
        }
      })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  });

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
    updatePreviewStream();
  }

  // Manual show/hide on top of the two automatic conditions below — lets
  // the operator stop the stream (save bandwidth/decode cost) without
  // having to leave the Capture tab or switch capture methods. Defaults
  // to shown; resets on page reload (not persisted, matching this
  // client's general "ephemeral UI state" precedent for things like the
  // Convert workspace selector).
  var previewHidden = false;
  var previewToggleBtn = document.getElementById("cap-preview-toggle-btn");
  previewToggleBtn.addEventListener("click", function () {
    previewHidden = !previewHidden;
    updatePreviewStream();
  });

  // Starts/stops the live capture-preview <img> (see GET /api/preview_stream
  // in web_control_server.py). Three independent conditions gate it, checked
  // fresh every call rather than cached: (1) screenshot_method must be one
  // of uvc/ndi/udp — mss/dxcam already show the desktop directly to whoever
  // is at that machine, streaming those adds nothing; (2) the Capture tab
  // must actually be the one on-screen (currentTab) — there's no reason to
  // keep decoding an MJPEG stream the operator isn't even looking at; (3)
  // the operator hasn't manually hidden it via the Hide/Show Preview button.
  // Setting/clearing img.src (rather than just hiding the element with CSS)
  // is what actually opens/closes the underlying connection — a hidden
  // <img> with a live src would keep pulling frames in the background.
  function updatePreviewStream() {
    var group = document.getElementById("cap-preview-group");
    var groupTitle = document.getElementById("cap-preview-group-title");
    var img = document.getElementById("cap-preview-img");
    var methodEl = document.getElementById("cap-screenshot_method");
    var method = methodEl ? methodEl.value : "";
    var isStream = method === "uvc" || method === "ndi" || method === "udp";
    group.classList.toggle("hidden", !isStream);
    groupTitle.classList.toggle("hidden", !isStream);
    previewToggleBtn.textContent = previewHidden ? "Show Preview" : "Hide Preview";
    img.classList.toggle("hidden", previewHidden);

    var shouldRun = isStream && currentTab === "capture" && !previewHidden;
    if (shouldRun) {
      if (!img.dataset.streaming) {
        img.src = "/api/preview_stream?preview_token=" + encodeURIComponent(getToken());
        img.dataset.streaming = "1";
      }
    } else if (img.dataset.streaming) {
      img.removeAttribute("src");
      delete img.dataset.streaming;
    }
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

  // Device/FPS selects have no data-key (their options are populated live
  // by the Refresh Device probe, not static HTML) — wire their writes
  // explicitly, same single-field pushTabSetting() every generic field uses.
  document.getElementById("cap-uvc-device-select").addEventListener("change", function (e) {
    var idx = parseInt(e.target.value, 10);
    if (isNaN(idx)) return;
    pushTabSetting("capture", "uvc_device_index", idx);
  });

  document.getElementById("cap-uvc-fps-select").addEventListener("change", function (e) {
    var fps = parseInt(e.target.value, 10);
    if (isNaN(fps)) return;
    pushTabSetting("capture", "uvc_fps", fps);
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

  // ---------------------------------------------------------------------
  // Visuals panel — Web ESP Enable is a dedicated action (not a plain
  // Config field: see web_control_settings.py's _SCHEMA["visuals"]
  // comment), since flipping it must actually start/stop the live
  // esp_server, mirroring visuals_page.py's _onWebEspEnableChanged(). Port
  // changes go through the generic field path (wireGenericFields("visuals"))
  // but additionally trigger a debounced restart so the already-running
  // server picks up the new port, mirroring _restartWebEspIfRunning()'s
  // 600ms QTimer.
  // ---------------------------------------------------------------------
  var suppressWebEspToggleEcho = false;
  var webEspRestartTimer = null;

  function applyVisualsExtras(data) {
    var toggle = document.getElementById("vis-web_esp_enabled");
    if (toggle !== document.activeElement) {
      suppressWebEspToggleEcho = true;
      toggle.checked = !!data.web_esp_enabled;
      suppressWebEspToggleEcho = false;
    }
    var urlEl = document.getElementById("vis-web-esp-url");
    urlEl.textContent = data.web_esp_running && data.web_esp_url ? data.web_esp_url : "—  (not running)";
  }

  function scheduleWebEspRestart() {
    if (webEspRestartTimer) clearTimeout(webEspRestartTimer);
    webEspRestartTimer = setTimeout(function () {
      webEspRestartTimer = null;
      fetch("/api/control/web_esp_restart", { method: "POST", headers: authHeaders() })
        .catch(function () {})
        .then(function () { loadTabSettings("visuals"); });
    }, 600);
  }

  document.getElementById("vis-web_esp_enabled").addEventListener("change", function (e) {
    if (suppressWebEspToggleEcho) return;
    var enabled = e.target.checked;
    e.target.disabled = true;
    fetch("/api/control/web_esp_enabled", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ enabled: enabled }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          e.target.checked = !enabled;
        } else {
          loadTabSettings("visuals");
        }
      })
      .catch(function () { e.target.checked = !enabled; })
      .then(function () { e.target.disabled = false; });
  });

  // Opens on the HOST machine running Axiom, not this remote browser tab
  // — same caveat as Model panel's Open Model Folder button.
  document.getElementById("vis-web-esp-open-btn").addEventListener("click", function () {
    var btn = document.getElementById("vis-web-esp-open-btn");
    btn.disabled = true;
    fetch("/api/control/web_esp_open", { method: "POST", headers: authHeaders() })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  });

  // ---------------------------------------------------------------------
  // Configs panel — preset CRUD via ConfigManager, plus content-based
  // Export/Import since a browser can't supply a host-side file path (see
  // web_control_settings.py's export_config_preset_content()/
  // import_config_preset_content() docstrings). window.prompt()/confirm()
  // stand in for configs_page.py's QInputDialog/QMessageBox — the direct
  // browser-native equivalent for this plain, no-framework client.
  // ---------------------------------------------------------------------
  var configsSelect = document.getElementById("configs-select");
  var configsReason = document.getElementById("configs-reason");

  function refreshConfigsList() {
    fetch("/api/configs", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || !data.presets) return;
        var current = configsSelect.value;
        configsSelect.innerHTML = "";
        data.presets.forEach(function (name) {
          var opt = document.createElement("option");
          opt.value = name;
          opt.textContent = name;
          configsSelect.appendChild(opt);
        });
        var has = Array.prototype.some.call(configsSelect.options, function (o) { return o.value === current; });
        if (has) configsSelect.value = current;
        // The Quick Presets sidebar section reuses this exact fetch rather
        // than making its own — one master list, same "allModelNames"
        // precedent the Model/Convert tabs already share.
        populatePresetSlotSelects(data.presets);
      })
      .catch(function () {});
  }

  document.getElementById("configs-refresh-btn").addEventListener("click", refreshConfigsList);

  document.getElementById("configs-create-btn").addEventListener("click", function () {
    var name = window.prompt("New preset name:");
    if (!name) return;
    fetch("/api/configs/save", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ name: name }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          configsReason.textContent = (data && data.reason) || "failed to create";
        } else {
          configsReason.textContent = "created \"" + name + "\"";
          refreshConfigsList();
        }
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  document.getElementById("configs-save-btn").addEventListener("click", function () {
    var name = configsSelect.value;
    if (!name) { configsReason.textContent = "select a config first"; return; }
    if (!window.confirm('Overwrite "' + name + '" with the current live settings?')) return;
    fetch("/api/configs/save", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ name: name }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        configsReason.textContent = (data && data.ok) ? "saved \"" + name + "\"" : ((data && data.reason) || "failed to save");
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  document.getElementById("configs-load-btn").addEventListener("click", function () {
    var name = configsSelect.value;
    if (!name) { configsReason.textContent = "select a config first"; return; }
    // Dry-run preview first (mirrors configs_page.py's Load button — see
    // ConfigManager.preview_config_changes()) so the operator sees what
    // would actually change before it's applied, instead of silently
    // overwriting the live config.
    fetch("/api/configs/preview?name=" + encodeURIComponent(name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        var changes = data && data.changes;
        var proceed = true;
        if (changes && changes.length) {
          proceed = window.confirm('Loading "' + name + '" will change:\n\n' + changes.join("\n") + "\n\nProceed?");
        }
        // changes === [] (identical to current) or preview unavailable
        // (null) both fall through to loading directly, same as the Qt
        // page's own fallback behavior.
        if (!proceed) return;
        fetch("/api/configs/load", {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ name: name }),
        })
          .then(function (res) { return res.ok ? res.json() : null; })
          .then(function (loadData) {
            if (loadData && loadData.ok) {
              configsReason.textContent = "loaded \"" + name + "\"";
              // The just-loaded preset can touch fields on any tab —
              // resync every generic-schema tab's settings so the whole
              // client reflects the new live config, not just Configs.
              ["model", "capture", "inference", "aim", "keys", "visuals"].forEach(loadTabSettings);
            } else {
              configsReason.textContent = (loadData && loadData.reason) || "failed to load";
            }
          })
          .catch(function () { configsReason.textContent = "request failed"; });
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  document.getElementById("configs-rename-btn").addEventListener("click", function () {
    var oldName = configsSelect.value;
    if (!oldName) { configsReason.textContent = "select a config first"; return; }
    var newName = window.prompt("Rename \"" + oldName + "\" to:", oldName);
    if (!newName || newName === oldName) return;
    fetch("/api/configs/rename", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ old_name: oldName, new_name: newName }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (data && data.ok) {
          configsReason.textContent = "renamed to \"" + newName + "\"";
          refreshConfigsList();
        } else {
          configsReason.textContent = (data && data.reason) || "failed to rename";
        }
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  document.getElementById("configs-delete-btn").addEventListener("click", function () {
    var name = configsSelect.value;
    if (!name) { configsReason.textContent = "select a config first"; return; }
    if (!window.confirm('Delete "' + name + '"? This cannot be undone.')) return;
    fetch("/api/configs/delete", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ name: name }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (data && data.ok) {
          configsReason.textContent = "deleted \"" + name + "\"";
          refreshConfigsList();
        } else {
          configsReason.textContent = (data && data.reason) || "failed to delete";
        }
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  // Export downloads the preset's raw JSON via a Blob + temporary <a
  // download> link — this page is the app's own hosted static client, not
  // a sandboxed Artifact, so a download link works normally here.
  document.getElementById("configs-export-btn").addEventListener("click", function () {
    var name = configsSelect.value;
    if (!name) { configsReason.textContent = "select a config first"; return; }
    fetch("/api/configs/export?name=" + encodeURIComponent(name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          configsReason.textContent = (data && data.reason) || "failed to export";
          return;
        }
        var blob = new Blob([data.content], { type: "application/json" });
        var url = URL.createObjectURL(blob);
        var a = document.createElement("a");
        a.href = url;
        a.download = name + ".json";
        document.body.appendChild(a);
        a.click();
        document.body.removeChild(a);
        URL.revokeObjectURL(url);
        configsReason.textContent = "exported \"" + name + "\"";
      })
      .catch(function () { configsReason.textContent = "request failed"; });
  });

  // Import reads the chosen file's content client-side (FileReader) and
  // POSTs it as JSON text — there's no host-side path a browser could send
  // instead, see import_config_preset_content()'s docstring.
  var configsImportFile = document.getElementById("configs-import-file");
  document.getElementById("configs-import-btn").addEventListener("click", function () {
    configsImportFile.click();
  });

  configsImportFile.addEventListener("change", function () {
    var file = configsImportFile.files && configsImportFile.files[0];
    configsImportFile.value = ""; // allow re-selecting the same file later
    if (!file) return;
    var reader = new FileReader();
    reader.onload = function () {
      fetch("/api/configs/import", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ content: reader.result }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (data && data.ok) {
            configsReason.textContent = "imported as \"" + data.name + "\"";
            refreshConfigsList();
          } else {
            configsReason.textContent = (data && data.reason) || "failed to import";
          }
        })
        .catch(function () { configsReason.textContent = "request failed"; });
    };
    reader.onerror = function () { configsReason.textContent = "failed to read file"; };
    reader.readAsText(file);
  });

  // Opens on the HOST machine running Axiom, not this remote browser tab
  // — same caveat as Model panel's Open Model Folder button.
  document.getElementById("configs-open-folder-btn").addEventListener("click", function () {
    var btn = document.getElementById("configs-open-folder-btn");
    btn.disabled = true;
    fetch("/api/control/open_configs_folder", { method: "POST", headers: authHeaders() })
      .catch(function () {})
      .then(function () { btn.disabled = false; });
  });

  // ---------------------------------------------------------------------
  // Quick Presets — 5 one-click shortcuts below the sidebar's Status
  // section, always visible regardless of which tab is open (unlike the
  // Configs panel itself). Each slot is independently assignable to a
  // saved preset via its own <select>; picking a value there POSTs the
  // assignment immediately (GET/POST /api/preset_slots), while Load fires
  // a bare POST /api/configs/load with no confirmation prompt — a
  // deliberate fast-path, unlike the Configs tab's own Load button, which
  // previews the diff first. Reuses refreshConfigsList()'s own fetched
  // preset list (populatePresetSlotSelects(), called from inside it) —
  // same "share one fetched list" precedent as allModelNames.
  // ---------------------------------------------------------------------
  var PRESET_SLOT_COUNT = 5;
  var presetSlotSelects = [];
  var presetSlotLoadBtns = [];
  for (var _psi = 0; _psi < PRESET_SLOT_COUNT; _psi++) {
    presetSlotSelects.push(document.getElementById("preset-slot-" + _psi));
    presetSlotLoadBtns.push(document.getElementById("preset-slot-load-" + _psi));
  }

  function populatePresetSlotSelects(names) {
    presetSlotSelects.forEach(function (sel) {
      if (document.activeElement === sel) return; // don't clobber a mid-pick
      var current = sel.value;
      sel.innerHTML = '<option value="">— unassigned —</option>';
      (names || []).forEach(function (name) {
        var opt = document.createElement("option");
        opt.value = name;
        opt.textContent = name;
        sel.appendChild(opt);
      });
      var has = Array.prototype.some.call(sel.options, function (o) { return o.value === current; });
      sel.value = has ? current : "";
    });
  }

  function fetchPresetSlotAssignments() {
    fetch("/api/preset_slots", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || !data.slots) return;
        data.slots.forEach(function (name, i) {
          var sel = presetSlotSelects[i];
          if (!sel || document.activeElement === sel) return;
          sel.value = name || "";
          presetSlotLoadBtns[i].disabled = !name;
        });
      })
      .catch(function () {});
  }

  presetSlotSelects.forEach(function (sel, i) {
    sel.addEventListener("change", function () {
      var name = sel.value;
      fetch("/api/preset_slots", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ index: i, name: name }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (data && data.ok) {
            presetSlotLoadBtns[i].disabled = !name;
          } else {
            // Refused (e.g. the picked name no longer exists) — resync
            // this row from the server's actual current assignment rather
            // than trusting the now-stale local selection.
            fetchPresetSlotAssignments();
          }
        })
        .catch(function () {});
    });
  });

  presetSlotLoadBtns.forEach(function (btn, i) {
    btn.addEventListener("click", function () {
      var name = presetSlotSelects[i].value;
      if (!name) return;
      btn.disabled = true;
      fetch("/api/configs/load", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ name: name }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (data && data.ok) {
            // Same resync every other successful preset load already does
            // — a preset can touch fields on any generic-schema tab.
            ["model", "capture", "inference", "aim", "keys", "visuals"].forEach(loadTabSettings);
          }
        })
        .catch(function () {})
        .then(function () { btn.disabled = !presetSlotSelects[i].value; });
    });
  });

  // ---------------------------------------------------------------------
  // Convert panel — kicks off a background TensorRT engine build and
  // polls its status/log, mirroring convert_page.py's _ConvertWorker (a
  // QThread + subprocess there; a plain background thread + subprocess on
  // the server here — see app_controller.start_conversion()). The model
  // to build and the workspace budget are one-shot POST parameters, never
  // persisted Config fields, matching convert_page.py's own workspaceCombo
  // (also never written back to Config). trt_fp16_enabled is the one
  // field that DOES round-trip through the generic settings mechanism —
  // marked data-custom="1" so it's read generically (pre-populates the
  // toggle from the last-used value, mirroring ConvertPage.setConfig())
  // but never auto-pushed on a bare flip; the real write happens only
  // after a successful build, exactly when _onConvertFinished() does it.
  // ---------------------------------------------------------------------
  var convertModelSelect = document.getElementById("convert-model-select");
  var convertFp16Toggle = document.getElementById("convert-trt_fp16_enabled");
  var convertWorkspaceSelect = document.getElementById("convert-workspace-select");
  var convertBtn = document.getElementById("convert-btn");
  var convertProgress = document.getElementById("convert-progress");
  var convertLog = document.getElementById("convert-log");
  var convertReason = document.getElementById("convert-reason");
  var convertModelListPopulated = false;
  var convertLogSince = 0;
  var convertPolling = false;
  // True whenever a build might still be in progress and pollConvertStatus()
  // should keep rescheduling itself — tracked separately from the poll's own
  // response so a single transient failure (network blip, momentary 401)
  // can never permanently wedge the follow loop (see pollConvertStatus()'s
  // own comment for the bug this replaced: an early `if (!data) return`
  // used to leave convertPolling stuck true forever, silently killing all
  // log-following for the rest of the page's life after just one failed
  // poll — this is why "no logs ever showed up" for the reported case).
  var convertWatching = false;

  function populateConvertModelSelect() {
    var current = convertModelSelect.value || modelSelect.value;
    convertModelSelect.innerHTML = "";
    allModelNames.forEach(function (name) {
      var opt = document.createElement("option");
      opt.value = name;
      opt.textContent = name;
      convertModelSelect.appendChild(opt);
    });
    var hasCurrent = Array.prototype.some.call(convertModelSelect.options, function (o) { return o.value === current; });
    if (hasCurrent) convertModelSelect.value = current;
  }

  function ensureConvertPanelExtras() {
    if (allModelNames.length) {
      // Re-populate on every activation (not just once) — allModelNames
      // may have grown since the last visit (loadModelList() only ever
      // fetches once, but a new .onnx file added while this page is open
      // still needs its own path to show up here eventually via a reload).
      populateConvertModelSelect();
      convertModelListPopulated = true;
    } else if (!convertModelListPopulated) {
      // The Model tab's own GET /api/models hasn't resolved yet (very
      // first page load, arriving here before it does) — try again
      // shortly rather than leaving the select permanently empty.
      setTimeout(ensureConvertPanelExtras, 300);
    }
    // Resume log-following if a build is already in progress — started
    // from the Qt GUI, from a previous visit to this tab, or still running
    // across a page reload (the server-side job isn't tied to any one
    // browser tab). Presumed true until the very first response proves
    // otherwise, at which point pollConvertStatus() itself corrects it
    // back to false and the chain stops after just one check.
    convertWatching = true;
    pollConvertStatus();
  }

  function setConvertRunningUi(running) {
    convertBtn.disabled = running;
    convertBtn.textContent = running ? "Converting…" : "Convert";
    convertModelSelect.disabled = running;
    convertWorkspaceSelect.disabled = running;
    convertProgress.classList.toggle("hidden", !running);
  }

  function pollConvertStatus() {
    if (convertPolling) return; // an identical in-flight request already covers this tick
    convertPolling = true;
    fetch("/api/convert/status?since=" + convertLogSince, { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) return; // transport-level failure — convertWatching is left as-is, see below
        convertLogSince = data.next_since;
        if (data.log_lines && data.log_lines.length) {
          data.log_lines.forEach(function (line) {
            convertLog.value += (convertLog.value ? "\n" : "") + line;
          });
          convertLog.scrollTop = convertLog.scrollHeight;
        }
        setConvertRunningUi(data.running);
        convertWatching = data.running;
        if (data.done && data.message) {
          convertReason.textContent = data.success
            ? "✓ Done. Engine cache written to: " + data.message
            : "✗ " + data.message;
        }
      })
      // A transient failure (network blip, a momentary bad-token response)
      // must never permanently stop this loop — convertWatching keeps
      // whatever value it already had, so the next tick just retries
      // instead of silently giving up on log-following forever.
      .catch(function () {})
      .then(function () {
        convertPolling = false;
        // Keep following the log every ~1s for as long as a build might
        // still be running — independent of which tab is on-screen, so a
        // build finishes and its result is ready even if the operator
        // wandered off to another tab in the meantime.
        if (convertWatching) setTimeout(pollConvertStatus, 1000);
      });
  }

  // Shared by the Convert button's own click and by the Model tab's
  // needs_conversion auto-redirect below — factored out so both start a
  // build the exact same way rather than duplicating the request/response
  // handling twice. `name` is a bare basename (convertModelSelect's own
  // option values); fp16/workspaceMb read the Convert panel's own current
  // toggle/select state when omitted, same values a plain button click
  // would have used.
  function startConversionFlow(name, fp16, workspaceMb) {
    if (!name) { convertReason.textContent = "select a model first"; return; }
    if (fp16 === undefined) fp16 = convertFp16Toggle.checked;
    if (workspaceMb === undefined) workspaceMb = parseInt(convertWorkspaceSelect.value, 10) || 2048;
    convertLog.value = "";
    convertLogSince = 0;
    convertReason.textContent = "";
    fetch("/api/control/convert", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        // resolve_model_path() joins a relative path directly against
        // project_root, not project_root/Model — same "Model/" prefix
        // the Model tab's own Switch button already adds.
        model_path: "Model/" + name,
        fp16: fp16,
        workspace_mb: workspaceMb,
      }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          convertReason.textContent = (data && data.reason) || "failed to start";
          return;
        }
        setConvertRunningUi(true);
        convertWatching = true;
        pollConvertStatus();
      })
      .catch(function () { convertReason.textContent = "request failed"; });
  }

  convertBtn.addEventListener("click", function () {
    startConversionFlow(convertModelSelect.value);
  });

  // The backend select offers only 4 options with no separate "CUDA" entry
  // (mirrors model_page.py's own reverse-map fold, which folds "cuda" onto
  // the "TensorRT" display item) — a "cuda" backend from the server is
  // shown as "TensorRT" here, but never written back as "cuda" in a POST
  // body, matching model_page.py's write-side map which also never writes
  // "cuda". Driven by s.selected_backend (config.inference_backend, the
  // user's persisted choice — "auto"/"tensorrt"/"cuda"/"directml"/"cpu"),
  // never s.inference_backend (the live ONNX EP string like
  // "TensorrtExecutionProvider" used for the plain-text status readout) —
  // that string never matches any option value here, which was leaving
  // this select stuck on its default "Auto" even while TensorRT was
  // actually active, exactly the way model_page.py's own combo tracks
  // config.inference_backend rather than the live resolved provider.
  function displayBackend(name) {
    return name === "cuda" ? "tensorrt" : name;
  }

  // Only the three capture backends that read from an external device/
  // stream rather than the desktop itself get a Stream FPS stat — mirrors
  // status_panel.py's own source_fps_row label map and unit convention
  // (fps for udp's real assembled-frame rate, Hz for uvc/ndi's device-
  // reported nominal rate). mss/dxcam get no row at all, same as this
  // client's own scoping request.
  var STREAM_FPS_LABELS = { uvc: "UVC Source FPS", ndi: "NDI Source FPS", udp: "UDP Stream FPS" };

  function applyStatus(s) {
    document.getElementById("s-running").textContent = s.running ? "RUNNING" : "stopped";
    document.getElementById("s-active").textContent = s.active ? "ON" : "OFF";
    document.getElementById("s-firing").textContent = s.aim_firing ? "ON" : "OFF";
    document.getElementById("s-model").textContent = s.model || "—";
    document.getElementById("s-backend").textContent = s.inference_backend || "—";
    document.getElementById("s-mouse").textContent = s.mouse_move_method || "—";
    document.getElementById("s-capfps").textContent = s.capture_fps != null ? s.capture_fps.toFixed(1) : "—";
    document.getElementById("s-inffps").textContent = s.inference_fps != null ? s.inference_fps.toFixed(1) : "—";

    var streamStat = document.getElementById("s-stream-fps-stat");
    var streamLabel = STREAM_FPS_LABELS[s.screenshot_method];
    if (streamLabel) {
      streamStat.classList.remove("hidden");
      document.getElementById("s-stream-fps-label").textContent = streamLabel;
      var isUdp = s.screenshot_method === "udp";
      var nominal = isUdp ? s.udp_recv_fps : s.source_fps;
      var unit = isUdp ? "fps" : "Hz";
      var streamText = nominal > 0 ? nominal.toFixed(0) + " " + unit : "—";
      if (isUdp && s.udp_dropped_fps >= 1.0) {
        streamText += "  ⚠ " + s.udp_dropped_fps.toFixed(0) + " dropped/s";
      }
      document.getElementById("s-stream-fps").textContent = streamText;
    } else {
      streamStat.classList.add("hidden");
    }

    suppressToggleEcho = true;
    alwaysAimToggle.checked = !!s.always_aim;
    suppressToggleEcho = false;

    // Keep the always_aim-gated Keys & HW cards live-synced from every
    // status poll (not just on tab activation), mirroring the Qt app's own
    // continuous re-check pattern (_updateMakcuAimStatus()) — this way a
    // toggle flipped from either the Qt GUI or this same page updates the
    // Keys panel even while it isn't the active tab.
    var hideForAlwaysAim = !!s.always_aim;
    document.getElementById("keys-makcu-trigger-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-aim-mode-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-disengage-card").classList.toggle("hidden", hideForAlwaysAim);

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
    // Two guards, both required — without either, the operator picking
    // "DirectML" here visibly snaps back to "TensorRT" within ~1s (this was
    // a real, reported bug): (1) document.activeElement !== backendSelect
    // skips the write while the dropdown is focused/open — the operator
    // hasn't clicked Switch yet, s.selected_backend still reflects the OLD
    // backend, and the very next poll tick (every ~1s) would otherwise
    // silently revert their in-progress choice before they can even click
    // Switch. (2) !modelSwitchPending skips it for the entire in-flight
    // switch flow — the initial POST's round trip, PLUS (for a
    // needs_restart refusal) the confirm() dialog and the subsequent
    // /api/control/model_restart round trip — during which
    // s.selected_backend is still the pre-switch value until that whole
    // flow actually lands, so reflecting it early would revert the
    // dropdown to the old backend right as the operator is confirming the
    // restart. See modelSwitchPending's own comment for exactly where it's
    // set/cleared.
    if (s.selected_backend && document.activeElement !== backendSelect && !modelSwitchPending) {
      var backendDisplay = displayBackend(s.selected_backend);
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

    // Keep whichever generic-schema tab is on-screen synced to the live
    // Config every tick too — not just on activation — so a change made
    // from the Qt GUI while this page is sitting open on that tab surfaces
    // within ~1s with no reload. loadTabSettings()'s own guards (skip a
    // focused field, skip a field whose paired range slider is mid-drag)
    // already keep this from fighting an in-progress edit here.
    if (GENERIC_SCHEMA_TABS.indexOf(currentTab) !== -1) {
      loadTabSettings(currentTab);
    }
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

  // request_model_change()'s refusal reasons are plain machine-readable
  // codes (see app_controller.py's own docstring for the full list) —
  // spelled out here so a refusal reads as an explained limitation rather
  // than a bare unexplained string. needs_restart mirrors a real,
  // unavoidable ONNX Runtime constraint: DirectML sessions can't be
  // hot-swapped, so model_page.py's own _onInferenceBackendChanged()
  // handles this locally by restarting the whole app
  // (_startRestartCountdown()) — this entry's text is only shown as a
  // fallback (offerModelRestartConfirm() below intercepts the real
  // needs_restart flow with a confirm dialog + an actual remote restart,
  // via confirm_model_change_with_restart()/POST /api/control/model_restart).
  var MODEL_SWITCH_REASONS = {
    no_model_path: "no model selected",
    invalid_model_path: "not a .onnx file",
    not_found: "model file not found on the host machine",
    invalid_backend: "unknown backend",
    needs_restart: "crossing to/from DirectML needs a full app restart",
    needs_conversion: "this model/backend combination needs a TensorRT build first — " +
      "use the Convert tab, then switch again",
  };

  function describeModelSwitchReason(reason) {
    return MODEL_SWITCH_REASONS[reason] || reason || "failed";
  }

  // Guards backendSelect (see applyStatus()) against a poll tick reverting
  // it while a switch is genuinely in flight — the plain round trip to
  // /api/control/model, PLUS, for a needs_restart refusal, the confirm()
  // dialog and the subsequent /api/control/model_restart round trip.
  // Deliberately NOT tied to modelSwitchBtn.disabled, which re-enables
  // right after the FIRST request settles (so the operator isn't stuck if
  // the confirm dialog takes a while to answer) — this flag instead spans
  // the operator's whole intent, cleared only once the flow actually
  // resolves one way or another (applied, refused-and-shown, declined, or
  // the restart request itself failed).
  var modelSwitchPending = false;

  // Confirmed-restart flow for the one reason above that a plain retry
  // can never fix: crossing to/from DirectML needs a real app restart
  // (see MODEL_SWITCH_REASONS.needs_restart's own text and
  // confirm_model_change_with_restart()'s docstring in app_controller.py).
  // window.confirm() is the direct browser-native equivalent of the Qt
  // app's own restart confirmation — nothing is restarted without it.
  function offerModelRestartConfirm(modelPath, inferenceBackend) {
    var confirmed = window.confirm(
      "Crossing to/from DirectML can't be hot-swapped — Axiom needs to fully " +
      "restart to apply this (the Qt app does the same restart locally for " +
      "this exact change).\n\n" +
      "Restart Axiom now? The app — and this page's connection to it — will " +
      "drop for a few seconds, then come back on its own."
    );
    if (!confirmed) {
      modelSwitchReason.textContent = "restart declined — model/backend left unchanged";
      modelSwitchPending = false;
      return;
    }
    modelSwitchReason.textContent = "restarting Axiom…";
    fetch("/api/control/model_restart", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ model_path: modelPath, inference_backend: inferenceBackend }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          modelSwitchReason.textContent = describeModelSwitchReason(data && data.reason);
          // Nothing was actually applied — safe (and correct) to let the
          // next poll resync backendSelect from the still-unchanged config.
          modelSwitchPending = false;
          return;
        }
        modelSwitchReason.textContent = "restarting Axiom — this page reconnects automatically once it's back up";
        // config.model_path/inference_backend are already written
        // server-side at this point (confirm_model_change_with_restart()
        // applies them before scheduling the restart) — releasing the
        // guard now is what lets applyStatus() start reflecting the real,
        // just-confirmed backend instead of fighting it for the remaining
        // few seconds until the process actually restarts.
        modelSwitchPending = false;
      })
      .catch(function () {
        // The process may already be mid-restart by the time this settles
        // (the response itself can race the exit) — poll()'s own retry
        // loop silently recovers once the new process's server responds
        // again, so there's nothing more to surface here than what's
        // already showing.
        modelSwitchPending = false;
      });
  }

  modelSwitchBtn.addEventListener("click", function () {
    // modelSelect's options are bare basenames (see loadModelList()) —
    // resolve_model_path() joins a relative path directly against
    // project_root, not project_root/Model, so the "Model/" prefix has to
    // be added client-side.
    var modelPath = "Model/" + modelSelect.value;
    modelSwitchPending = true;
    modelSwitchBtn.disabled = true;
    modelSwitchReason.textContent = "";
    fetch("/api/control/model", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({ model_path: modelPath, inference_backend: backendSelect.value }),
    })
      .then(function (res) {
        if (!res.ok) {
          // Show the actual status instead of a bare "request failed" —
          // 401 (bad/stale token), 422 (malformed body), and a 5xx server
          // error all need a different fix, and a generic message can't
          // be told apart from the network-failure case in .catch() below.
          return res.text().then(function (bodyText) {
            var detail = bodyText ? " — " + bodyText.slice(0, 200) : "";
            modelSwitchReason.textContent = "request failed (HTTP " + res.status + ")" + detail;
            return null;
          }, function () {
            modelSwitchReason.textContent = "request failed (HTTP " + res.status + ")";
            return null;
          });
        }
        return res.json();
      })
      .then(function (data) {
        if (!data) {
          // The !res.ok branch above already reported its own reason text
          // and resolved with null — nothing was applied, so it's safe
          // (and necessary — otherwise this would wedge backendSelect's
          // poll-guard permanently true) to release the guard here too.
          modelSwitchPending = false;
          return;
        }
        if (data.ok === false) {
          if (data.reason === "needs_restart") {
            // offerModelRestartConfirm() owns clearing modelSwitchPending
            // itself once its own longer flow (confirm dialog + possible
            // /api/control/model_restart round trip) actually settles —
            // see its own comment. Skip the generic clear below for this
            // one branch only.
            offerModelRestartConfirm(modelPath, backendSelect.value);
            return;
          }
          if (data.reason === "needs_conversion") {
            // Mirrors model_page.py's own _redirectToConvertIfNeeded() —
            // rather than just telling the operator to go build it
            // themselves on the Convert tab, redirect there automatically
            // and start the build immediately, exactly like the Qt app
            // already does locally for this same case. Once the build
            // succeeds, _run_conversion_worker() (app_controller.py)
            // itself already applies + saves config.model_path — no
            // separate follow-up "switch" call is needed here, same as
            // the local Qt flow's own _onConvertFinished().
            modelSwitchReason.textContent = "needs a TensorRT build first — redirecting to Convert…";
            var bareName = modelSelect.value;
            activateTab("convert");
            convertModelSelect.value = bareName;
            startConversionFlow(bareName);
          } else {
            modelSwitchReason.textContent = describeModelSwitchReason(data.reason);
          }
        } else {
          // Never optimistically write s-model/s-backend here — the next
          // status poll is the sole source of truth for those, same as
          // every other field this client doesn't own a checkbox for.
          modelSwitchReason.textContent = data.applied_live
            ? "applied — live"
            : "applied — takes effect on next AI start";
        }
        modelSwitchPending = false;
      })
      .catch(function (err) {
        // A thrown exception here means the request never got a response
        // at all (network error, CORS, server unreachable) — distinct
        // from the !res.ok branch above, which did get a response.
        modelSwitchReason.textContent = "request failed (" + (err && err.message ? err.message : "network error") + ")";
        modelSwitchPending = false;
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
  wireGenericFields("aim");
  wireGenericFields("keys");
  wireGenericFields("visuals");
  wireGenericFields("trigger");
  wireGenericFields("convert");

  // A one-time progressive-enhancement pass over every numeric field in
  // the document (all panels are already in the DOM, just hidden by CSS
  // until activated — see the .panel/.tab comment near the top of this
  // file), not tied to any one tab's activation.
  enhanceNumberInputsWithSliders();

  loadModelList();
  // Model is the default-active panel (no click to trigger activateTab's
  // own load), so its tab settings + HUD extras are fetched explicitly here.
  loadTabSettings("model");
  ensureModelPanelExtras();
  // Quick Presets lives in the sidebar, visible regardless of which tab is
  // open — fetch its preset list + slot assignments once at startup too,
  // not gated on the Configs tab ever being activated.
  refreshConfigsList();
  fetchPresetSlotAssignments();
  poll();
  setInterval(poll, POLL_MS);
})();
