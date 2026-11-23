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
  var vigemNetToggle = document.getElementById("keys-vigem-net-toggle");
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

  // Generic per-viewer localStorage helpers, same "wrap every access in
  // try/catch" posture as getToken()/setToken() above (private browsing,
  // cleared site data, or storage-blocking browser settings must never
  // break the page — they just mean nothing persisted this time). Used
  // below to remember which tab was last open and the Quick Presets
  // sidebar's last-known state, so a page reload/browser reset doesn't
  // visually reset to defaults while waiting on the network — the real
  // source of truth is always the server; this is only a fast local cache
  // for the instant a fresh page starts rendering.
  function getLocalJSON(key) {
    try {
      var raw = localStorage.getItem(key);
      return raw ? JSON.parse(raw) : null;
    } catch (e) {
      return null;
    }
  }

  function setLocalJSON(key, value) {
    try {
      localStorage.setItem(key, JSON.stringify(value));
    } catch (e) {
      // Ignored — see getLocalJSON()'s own comment.
    }
  }

  var LAST_TAB_KEY = "axiom_last_tab";
  var PRESET_NAMES_CACHE_KEY = "axiom_preset_names_cache";
  var PRESET_SLOTS_CACHE_KEY = "axiom_preset_slots_cache";

  tokenInput.value = getToken();

  tokenSaveBtn.addEventListener("click", function () {
    setToken(tokenInput.value.trim());
    authState = "unknown";
    pollFailures = 0;
    tokenInput.classList.remove("attn");
    notify(tokenInput.value.trim() ? "Token saved — checking…" : "Token cleared", !tokenInput.value.trim());
    // A wrong/empty token on first load leaves the model <select> empty with
    // no retry — re-fetch models (and related one-shot lists) now that auth
    // may actually succeed, instead of requiring a full page reload.
    modelListLoaded = false;
    hudModelsLoaded = false;
    gameProfilesLoaded = false;
    loadModelList();
    if (typeof ensureModelPanelExtras === "function") ensureModelPanelExtras();
    if (typeof ensureVkOptionsLoaded === "function") ensureVkOptionsLoaded();
    poll();
  });

  tokenShowBtn.addEventListener("click", function () {
    tokenInput.type = tokenInput.type === "password" ? "text" : "password";
    tokenShowBtn.setAttribute("aria-pressed", tokenInput.type === "text" ? "true" : "false");
  });

  // "connected" | "disconnected" | "warn" (needs attention, e.g. no token yet).
  // `detail` becomes the badge's tooltip (HTTP status, error text, ...).
  function setStatusBadge(connected, label, detail, warn) {
    statusEl.textContent = label || (connected ? "connected" : "disconnected");
    statusEl.className = connected ? "connected" : (warn ? "warn" : "disconnected");
    if (detail) statusEl.title = detail; else statusEl.removeAttribute("title");
  }

  // Toast — transient feedback for saves/failures that used to be silent.
  // Identical messages are collapsed while one is still on screen so a
  // slider drag (many change events) or a failing 1-3s sync can't stack
  // dozens of toasts.
  var toastRegion = document.getElementById("toast-region");
  function notify(msg, isError, ms) {
    if (!toastRegion || !msg) return;
    var existing = toastRegion.children;
    for (var i = 0; i < existing.length; i++) {
      if (existing[i].textContent === msg) return;
    }
    var el = document.createElement("div");
    el.className = "toast" + (isError ? " error" : "");
    el.textContent = msg;
    toastRegion.appendChild(el);
    setTimeout(function () {
      if (el.parentNode) el.parentNode.removeChild(el);
    }, ms || (isError ? 5000 : 1600));
  }

  // Connection/auth state shared by poll() and api().
  var authState = "unknown"; // "unknown" | "ok" | "bad"
  var pollFailures = 0;

  // One place for auth header, 401, non-OK status and network errors.
  // Resolves to parsed JSON; rejects with an Error whose message is already
  // user-presentable (and `.status` set for HTTP failures).
  function api(method, path, body) {
    var opts = { method: method, headers: authHeaders() };
    if (body !== undefined) opts.body = JSON.stringify(body);
    return fetch(path, opts).then(function (res) {
      if (res.status === 401) {
        // With no token at all poll() owns the "enter token" badge; only a
        // token the server actually rejected is "bad".
        if (getToken()) {
          authState = "bad";
          setStatusBadge(false, "bad token", "The server rejected the access token");
        }
        var e401 = new Error("Unauthorized — check the access token");
        e401.status = 401;
        throw e401;
      }
      if (!res.ok) {
        var e = new Error("Server error (HTTP " + res.status + ")");
        e.status = res.status;
        throw e;
      }
      return res.json();
    }, function () {
      throw new Error("Cannot reach the Axiom host");
    });
  }

  // One call for an action route: toasts the reason when the request fails or is refused (no more silent
  // failures) and resolves to the response body, or null on failure.
  function actionCall(label, method, path, body, okMessage) {
    return api(method, path, body)
      .then(function (data) {
        if (data && data.ok === false) {
          notify(label + ": " + String(data.reason || "refused").replace(/_/g, " "), true);
          return null;
        }
        if (okMessage) notify(okMessage);
        return data || {};
      })
      .catch(function (err) {
        notify(label + ": " + (err && err.message ? err.message : "request failed"), true);
        return null;
      });
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
    // Remembered client-side so the next page load (a plain refresh, or a
    // full browser reset that clears nothing but this origin's storage
    // state) reopens on the same tab instead of always starting back on
    // Model — see the restore call near the bottom of this file's startup
    // block.
    setLocalJSON(LAST_TAB_KEY, name);
    tabButtons.forEach(function (btn) {
      var on = btn.getAttribute("data-tab") === name;
      btn.classList.toggle("active", on);
      btn.setAttribute("aria-selected", on ? "true" : "false");
      btn.tabIndex = on ? 0 : -1;
      if (on && btn.scrollIntoView) {
        try { btn.scrollIntoView({ block: "nearest", inline: "center" }); } catch (e) { btn.scrollIntoView(); }
      }
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
      refreshFullConfigsList();
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
  var suppressVigemNetToggleEcho = false;

  // The model select is populated once from GET /api/models (loadModelList,
  // fetched once on page load — not every poll tick) and the "currently
  // active" entry is applied once from the first status poll, same one-shot
  // semantics the old text-input prefill had. The two requests can land in
  // either order, so pendingModelSelection holds the status poll's answer
  // if it arrives before the model list has options to select among.
  var modelFormPrefilled = false;
  var modelListLoaded = false;
  var pendingModelSelection = null;
  // Picks the operator has made in the Model/Precision/Backend selects but not applied yet. The poll
  // must never overwrite them (they are applied together by Switch); once they equal the server's value,
  // or Switch resolves one way or another, they sync from the host again — so a change made in the Qt app
  // shows up here too.
  var modelStaged = { model: false, precision: false, backend: false };
  function clearModelStaged() {
    modelStaged.model = modelStaged.precision = modelStaged.backend = false;
    updateModelPendingHint();
  }
  function updateModelPendingHint() {
    var el = document.getElementById("model-pending-hint");
    if (el) el.hidden = !(modelStaged.model || modelStaged.precision || modelStaged.backend);
  }
  function stageModelPick(kind) {
    modelStaged[kind] = true;
    updateModelPendingHint();
  }
  var allModelNames = []; // master list — modelSearchInput filters a view of this
  // {basename: already_has_a_cached_.engine} and "is TensorRT the active
  // backend right now" — both come from GET /api/models alongside `models`
  // itself (app_controller.get_model_cache_status(), mirroring
  // model_page.py's own _computeEngineCacheStatus()/_isTensorRtActive()).
  // Badges are meaningless noise under any other backend, same as the Qt
  // combo hiding its icons entirely in that case.
  var modelCacheStatus = {};
  var modelTrtActive = false;

  // ✓/⬇ prefix mirroring model_page.py's _applyModelBadges() (FluentIcon.ACCEPT /
  // FluentIcon.CLOUD_DOWNLOAD as per-item combo icons there) — a plain HTML
  // <option> has no icon slot, so the equivalent glyph is prepended to the
  // visible label instead. The <option>'s own value stays the bare filename
  // either way, so nothing downstream (selection matching, POST bodies) needs
  // to know about this cosmetic prefix.
  var modelCacheFp16 = {};
  var modelCacheFp32 = {};

  function selectedModelFp16() {
    var el = document.getElementById("model-precision-select");
    return !el || el.value === "fp16";
  }

  function modelIsCached(name) {
    if (selectedModelFp16()) {
      if (name in modelCacheFp16) return !!modelCacheFp16[name];
    } else if (name in modelCacheFp32) {
      return !!modelCacheFp32[name];
    }
    return !!modelCacheStatus[name];
  }

  function formatModelOptionLabel(name) {
    if (!modelTrtActive) return name;
    return (modelIsCached(name) ? "✓ " : "⬇ ") + name;
  }

  function refreshModelOptionLabels() {
    Array.prototype.forEach.call(modelSelect.options, function (opt) {
      opt.textContent = formatModelOptionLabel(opt.value);
    });
    updateModelCacheLegend();
  }

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
      if (modelSelect.value !== name) {
        modelSelect.value = name;
        refreshModelInfoAndNotes();
      }
    }
    // else: the active model isn't in Model/ anymore (renamed/deleted since
    // it was loaded) — leave the select on its default rather than adding
    // a phantom option for a file that may no longer exist.
  }

  function setModelListHint(msg, isError) {
    if (modelSwitchReason) {
      modelSwitchReason.textContent = msg || "";
      modelSwitchReason.className = isError ? "hint error" : "hint";
    }
    var emptyHint = document.getElementById("model-empty-hint");
    if (emptyHint) {
      emptyHint.hidden = !!(msg || (allModelNames && allModelNames.length));
    }
  }

  function loadModelList() {
    fetch("/api/models", { headers: authHeaders() })
      .then(function (res) {
        if (res.status === 401) {
          setModelListHint("Unauthorized — check the Access Token, then Save.", true);
          return null;
        }
        if (!res.ok) {
          setModelListHint("Could not load models (HTTP " + res.status + ").", true);
          return null;
        }
        return res.json();
      })
      .then(function (data) {
        if (!data || !data.models) {
          if (!modelSwitchReason || !modelSwitchReason.textContent) {
            setModelListHint("Could not load model list. Check the token and try Refresh.", true);
          }
          modelSwitchBtn.disabled = true;
          return;
        }
        // Preserve whatever's currently selected across the rebuild below —
        // this function is also called after a successful switch (to
        // refresh the ✓/⬇ badges for the newly-active backend), where
        // silently resetting the visible selection back to the first
        // option would look like the switch didn't actually take.
        var currentSelection = modelSelect.options.length ? modelSelect.value : "";
        allModelNames = data.models;
        modelCacheStatus = data.cached || {};
        modelCacheFp16 = data.cached_fp16 || {};
        modelCacheFp32 = data.cached_fp32 || {};
        modelTrtActive = !!data.trt_active;
        modelSelect.innerHTML = "";
        data.models.forEach(function (name) {
          var opt = document.createElement("option");
          opt.value = name;
          opt.textContent = formatModelOptionLabel(name);
          modelSelect.appendChild(opt);
        });
        if (currentSelection) modelSelect.value = currentSelection;
        updateModelCacheLegend();
        modelListLoaded = true;
        modelSwitchBtn.disabled = data.models.length === 0;
        var emptyHint = document.getElementById("model-empty-hint");
        if (emptyHint) emptyHint.hidden = data.models.length > 0;
        if (data.models.length === 0) {
          setModelListHint("No .onnx files in Model/ on the host.", true);
        } else if (modelSwitchReason && /Unauthorized|Could not load/i.test(modelSwitchReason.textContent)) {
          setModelListHint("", false);
        }
        if (pendingModelSelection) {
          applyModelSelection(pendingModelSelection);
          pendingModelSelection = null;
        }
      })
      .catch(function () {
        setModelListHint("Network error loading models. Retry with Refresh.", true);
        modelSwitchBtn.disabled = true;
      });
  }

  var modelRefreshBtn = document.getElementById("model-refresh-btn");
  if (modelRefreshBtn) {
    modelRefreshBtn.addEventListener("click", function () {
      loadModelList();          // keeps the current (possibly staged) selection; only the list is reloaded
    });
  }

  // Explains the ✓/⬇ prefix — only shown while it's actually meaningful,
  // mirroring model_page.py's own modelCard subtitle toggle
  // (_updateModelCardLegend()).
  function updateModelCacheLegend() {
    var el = document.getElementById("model-cache-legend");
    if (!el) return;
    var precSel = document.getElementById("model-precision-select");
    if (precSel) (precSel.closest(".dd") || precSel).classList.toggle("hidden", !modelTrtActive);
    if (modelTrtActive) {
      var precision = selectedModelFp16() ? "FP16" : "FP32";
      el.textContent = "✓ = TensorRT engine already cached  ·  ⬇ = needs a one-time build (1-5 min)  (" + precision + ")";
      el.hidden = false;
    } else {
      el.hidden = true;
    }
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
        opt.textContent = formatModelOptionLabel(name);
        modelSelect.appendChild(opt);
        if (name === current) matchedCurrent = true;
      }
    });
    if (!matchedCurrent && current) {
      var opt2 = document.createElement("option");
      opt2.value = current;
      opt2.textContent = formatModelOptionLabel(current);
      modelSelect.appendChild(opt2);
    }
    modelSelect.value = current;
  });

  // ---------------------------------------------------------------------
  // Model Info / Model Notes — mirrors model_page.py's live inspector and
  // per-model notes card. Refreshed whenever the model selection settles
  // (user pick, or the status-driven one-shot prefill above).
  // ---------------------------------------------------------------------
  // Just the info card (not the notes): re-run whenever the live session changes -- model
  // swap, FP32/FP16 switch, provider change -- so "Precision in use" never goes stale.
  function refreshModelInfo() {
    var name = modelSelect.value;
    var infoEl = document.getElementById("model-info-text");
    if (!name) { infoEl.textContent = "\u2014"; return; }
    // /api/model_info resolves via resolve_model_path(), which joins a
    // relative path directly against project_root, not project_root/Model
    // — same "Model/" prefix the Switch button already adds, needed here
    // too since modelSelect's options are bare basenames.
    fetch("/api/model_info?model=" + encodeURIComponent("Model/" + name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data) { infoEl.textContent = "\u2014"; return; }
        if (data.ok === false) {
          infoEl.textContent = "Unavailable (" + (data.reason || "error") + ")";
          return;
        }
        infoEl.textContent = data.text || "\u2014";
      })
      .catch(function () { infoEl.textContent = "\u2014"; });
  }

  var lastModelRuntimeSig = null;
  function noteModelRuntime(s) {
    var sig = [s.loaded_model, s.loaded_model_fp16, s.loaded_provider, s.trt_fp16_enabled, s.model].join("|");
    if (lastModelRuntimeSig !== null && sig !== lastModelRuntimeSig) refreshModelInfo();
    lastModelRuntimeSig = sig;
  }

  function refreshModelInfoAndNotes() {
    var name = modelSelect.value;
    var infoEl = document.getElementById("model-info-text");
    var notesEl = document.getElementById("model-notes-text");
    if (!name) {
      infoEl.textContent = "—";
      notesEl.value = "";
      return;
    }
    refreshModelInfo();

    fetch("/api/model_notes?model=" + encodeURIComponent(name), { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        notesEl.value = data && data.text ? data.text : "";
      })
      .catch(function () {});
  }

  modelSelect.addEventListener("change", function () {
    stageModelPick("model");
    refreshModelInfoAndNotes();
  });
  backendSelect.addEventListener("change", function () { stageModelPick("backend"); });

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
      actionCall("Could not save notes", "POST", "/api/model_notes", { model: name, text: modelNotesTextarea.value }, "Notes saved");
    }
  });

  var openModelFolderBtn = document.getElementById("open-model-folder-btn");
  openModelFolderBtn.addEventListener("click", function () {
    openModelFolderBtn.disabled = true;
    actionCall("Could not open the folder", "POST", "/api/control/open_model_folder", {}, "Opened on the Axiom PC")
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
  var SLIDER_EXCLUDED_KEYS = ["web_esp_http_port", "web_esp_ws_port", "makcu_udp_port"];

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
    if (type === "number") return parseFloat(el.value);
    return el.value;
  }

  function loadTabSettings(tab, silent) {
    api("GET", "/api/settings/" + tab)
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
        if (tab === "capture" || tab === "model") noteWeaponMode(data.second_inference_mode);
        if (tab === "inference") {
          updateFovReduceVisibility();
          updateAnchorVisibility();
          updateDetectModeVisibility();
          updateWeaponDetectVisibility(data.second_inference_mode);
          updateInferenceProviderVisibility(data);
        }
        if (tab === "aim") {
          updateHumanizationVisibility();
          updateAimTuningVisibility();
          if (typeof data.pid_unsafe_mode === "boolean") applyPidUnsafeSliderLimits(!!data.pid_unsafe_mode);
        }
        if (tab === "keys") {
          ensureOption(document.getElementById("keys-makcu_com_port"), data.makcu_com_port);
          if (data.makcu_com_port && document.activeElement !== document.getElementById("keys-makcu_com_port")) {
            document.getElementById("keys-makcu_com_port").value = data.makcu_com_port;
          }
          var vigemTok = document.getElementById("keys-vigem_net_token");
          if (vigemTok) vigemTok.placeholder = data.vigem_net_token_set ? "saved — type to replace" : "token";
          if (!serialPortsLoaded) refreshSerialPorts(false);
          updateKeysVisibility(data);
        }
        if (tab === "visuals") { applyVisualsExtras(data); updateBoxStyleVisibility(); updateVisualsContextVisibility(data); updateGpuMonitorVisibility(); }
        if (tab === "convert" && typeof convertPrecisionSelect !== "undefined" && convertPrecisionSelect
            && document.activeElement !== convertPrecisionSelect && typeof data.trt_fp16_enabled === "boolean") {
          convertPrecisionSelect.value = data.trt_fp16_enabled ? "fp16" : "fp32";
        }
      })
      .catch(function (err) {
        // Background syncs stay quiet (the status badge already shows the
        // connection state); a load the user just asked for (tab switch,
        // preset load) says why it failed.
        if (!silent && err && err.status !== 401) notify("Could not load " + tab + " settings: " + err.message, true);
      });
  }

  // A loaded preset/config can touch fields on any tab (and a model/precision change): re-read every
  // generic-schema tab plus the model list, so the whole client shows the new live config.
  function resyncAllTabs() {
    clearModelStaged();
    GENERIC_SCHEMA_TABS.forEach(function (t) { loadTabSettings(t, true); });
    loadModelList();
  }

  function pushTabSetting(tab, key, value) {
    var body = {};
    body[key] = value;
    api("POST", "/api/settings/" + tab, body)
      .then(function (data) {
        if (!data || data.ok === false) {
          // Rejected — say why, and resync this tab's real values instead
          // of leaving the UI showing an edit that never actually applied.
          var why = data && data.reason ? String(data.reason).replace(/_/g, " ") : "rejected";
          notify("Could not save " + key + ": " + why, true);
          loadTabSettings(tab, true);
        } else {
          notify("Saved");
        }
      })
      .catch(function (err) {
        notify("Could not save " + key + ": " + (err && err.message ? err.message : "request failed"), true);
        loadTabSettings(tab, true);
      });
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
        if (tab === "capture" && (key === "uvc_capture_method" || key === "uvc_engine")) {
          updateUvcSubVisibility();
        }
        if (tab === "inference" && (key === "fov_mode" || key === "fov_retention_enabled")) updateFovReduceVisibility();
        if (tab === "inference" && (key === "afov_source" || key === "afov_lost_behavior")) updateAnchorFovVisibility();
        if (tab === "inference" && (key === "anchor_enabled" || key === "fov_mode")) updateAnchorVisibility();
        if (tab === "inference" && key === "second_inference_mode") updateWeaponDetectVisibility(value);
        if (tab === "capture" && key === "hud_udp_enabled") updateWeaponDetectVisibility();
        if (tab === "inference" && key === "detect_mode") updateDetectModeVisibility();
        if (tab === "aim" && HUMANIZATION_TOGGLE_KEYS.indexOf(key) !== -1) updateHumanizationVisibility();
        if (tab === "aim" && AIM_TUNING_TOGGLE_KEYS.indexOf(key) !== -1) updateAimTuningVisibility();
        if (tab === "visuals" && key === "box_full_rect") updateBoxStyleVisibility();
        if (tab === "visuals" && LINE_THICKNESS_TOGGLE_KEYS.indexOf(key) !== -1) updateLineThicknessVisibility();
        if (tab === "visuals" && key === "status_panel_show_gpu") updateGpuMonitorVisibility();
        if (tab === "visuals" && key === "enable_acrylic") updateVisualsContextVisibility();
        if (tab === "visuals" && (key === "web_esp_http_port" || key === "web_esp_ws_port")) {
          scheduleWebEspRestart();
        }
        if (tab === "keys" && (key === "makcu_connection" || key === "makcu_device_hotkeys" || key === "makcu_mouse_spread_enabled" || key === "remote_btn_enabled")) {
          updateKeysVisibility(lastSettings.keys || {});
        }
        // Always Auto-Fire no longer rewrites the saved "When Not Aiming" choice: the loop already
        // ignores the reduced rate while it is on (core/detect_mode.py), and the Inference tab shows a note.
      });
    });
  }

  // Mirrors inference_page.py's _updateFovReduceVisibility()/_updateTrackingVisibility():
  // the shrink settings need a shrinking FOV mode (Fixed/Adaptive/Dynamic); Dynamic Safeguards only in
  // Dynamic; the retention cards need Sticky Target Lock too (the lock settings are on the Aim tab).
  function updateFovReduceVisibility() {
    var d = lastSettings.inference || {};
    var el = function (id) { return document.getElementById(id); };
    var mode = (el("inf-fov_mode") || {}).value || d.fov_mode || "off";
    var on = mode === "fixed" || mode === "adaptive" || mode === "dynamic";
    var sticky = !!d.sticky_lock_enabled;
    var ret = el("inf-fov_retention_enabled") ? el("inf-fov_retention_enabled").checked : !!d.fov_retention_enabled;
    var show = function (id, v) { var c = el(id); if (c) c.classList.toggle("hidden", !v); };
    // Anchor FOV owns the centre and radius: the shrink cards only apply to Fixed/Adaptive/Dynamic.
    var shrink = on;
    show("inf-fov-min-size-card", shrink);
    show("inf-fov-min-duration-card", shrink);
    show("inf-fov-mode-card", true);
    show("inf-fov-dynsafe-card", shrink && mode === "dynamic");
    show("inf-fov-recovery-card", shrink);
    show("inf-fov-retention-card", shrink && sticky);
    show("inf-fov-retention-pct-card", shrink && sticky && ret);
    updateAnchorFovVisibility();
    show("inf-sticky-hint-card", shrink && !sticky);
  }

  // Mirrors inference_page.py's _updateAnchorFovVisibility(): the Anchor FOV groups appear only
  // in Anchor mode; Position needs a box source (Manual is a point from the crosshair; the
  // offsets apply to every source).
  function updateAnchorFovVisibility() {
    var d = lastSettings.inference || {};
    var el = function (id) { return document.getElementById(id); };
    var val = function (key, fallback) { var e = el("inf-" + key); return e ? (e.type === "checkbox" ? e.checked : e.value) : (d[key] !== undefined ? d[key] : fallback); };
    var mode = val("fov_mode", "off");
    var anchor = mode === "anchor";
    var src = val("afov_source", "detection");
    var show = function (id, v) { var c = el(id); if (c) c.classList.toggle("hidden", !v); };
    ["anchor", "move", "lost"].forEach(function (k) { show("inf-afov-sec-" + k, anchor); });
    show("inf-afov-position-card", anchor && src !== "manual");
  }

  // Mirrors inference_page.py's _updateAnchorVisibility(): Anchor Box Prediction shows only what is
  // FOV-specific, and only while it acts -- the horizon and the estimator hint with the switch, the
  // influence of the current shrink mode only (Fixed and Anchor FOV have none). The estimator tuning
  // is on the Aim tab (updateAimTuningVisibility()).
  function updateAnchorVisibility() {
    var d = lastSettings.inference || {};
    var el = function (id) { return document.getElementById(id); };
    var on = el("inf-anchor_enabled") ? el("inf-anchor_enabled").checked : !!d.anchor_enabled;
    var fovMode = (el("inf-fov_mode") || {}).value || d.fov_mode || "off";
    var fov = fovMode === "fixed" || fovMode === "adaptive" || fovMode === "dynamic";
    var show = function (id, v) { var c = el(id); if (c) c.classList.toggle("hidden", !v); };
    show("inf-anchor-horizon-card", on);
    show("inf-anchor-hint-card", on);
    show("inf-anchor-infl-adaptive-card", on && fov && fovMode === "adaptive");
    show("inf-anchor-infl-dynamic-card", on && fov && fovMode === "dynamic");
  }

  // The note says when "Reduced speed" cannot apply (server extra reduced_rate_ignored = Always Aim /
  // Always Auto-Fire on), mirroring inference_page.py.
  function updateDetectModeVisibility() {
    var d = lastSettings.inference || {};
    var sel = document.getElementById("inf-detect_mode");
    var mode = sel ? sel.value : d.detect_mode;
    var note = document.getElementById("inf-detect-mode-note");
    if (note) note.classList.toggle("hidden", !(mode === "reduced" && d.reduced_rate_ignored));
  }

  // Same display rules as inference_page.py: the cursor-follow FOV is meaningless for the external feeds
  // (uvc/ndi/udp). Display only — the
  // saved values are untouched (the loop derives the effective ones).
  function updateInferenceProviderVisibility(data) {
    var follow = document.getElementById("inf-follow-mouse-card");
    var external = ["uvc", "ndi", "udp"].indexOf(String(data.screenshot_method || "").toLowerCase()) !== -1;
    if (follow) follow.classList.toggle("hidden", external);
  }

  // Weapon Detection method (Inference tab) -> what the other tabs show. Mirrors the Qt pages:
  // off = nothing weapon-related; OCR = scan rate + ROI; ONNX = + threshold, HUD model, HUD UDP stream.
  var weaponMode = "off";
  function noteWeaponMode(mode) {
    if (typeof mode === "string") weaponMode = mode;
    updateWeaponDetectVisibility();
  }
  function updateWeaponDetectVisibility(mode) {
    if (typeof mode === "string") weaponMode = mode;
    var onnx = weaponMode === "v2_onnx", active = onnx || weaponMode === "v1_ocr";
    function toggle(id, show) {
      var el = document.getElementById(id);
      if (el) el.classList.toggle("hidden", !show);
    }
    var hudUdp = document.getElementById("cap-hud_udp_enabled");
    toggle("cap-weapon-module", active);
    toggle("cap-hud-udp-enable-card", onnx);
    toggle("cap-hud-udp-port-card", onnx && !!(hudUdp && hudUdp.checked));
    toggle("model-hud-module", onnx);
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
  ];

  // Advanced Aim Tuning: each sub-setting is shown only while the option it
  // tunes is active, mirroring aim_page.py's _updateAimTuningVisibility().
  var AIM_TUNING_TOGGLE_KEYS = ["prediction_enabled", "prediction_method", "aim_ema_smoothing_enabled",
    "sticky_lock_enabled", "kalman_enabled", "cam_motion_comp_enabled", "aim_y_reduce_enabled",
    "target_priority_mode", "anchor_mode"];

  function updateAimTuningVisibility() {
    function el(id) { return document.getElementById(id); }
    function setCard(id, show) {
      var input = el(id);
      var card = input && input.closest(".field-card");
      if (card) card.classList.toggle("hidden", !show);
    }
    var predOn = !!(el("aim-prediction_enabled") && el("aim-prediction_enabled").checked);
    var method = el("aim-prediction_method") ? el("aim-prediction_method").value : "velocity";
    setCard("aim-prediction_method", predOn);
    setCard("aim-prediction_adaptive_lead", predOn);
    setCard("aim-prediction_capture_timestamps", predOn);
    setCard("aim-prediction_lsq_fit", predOn && method === "velocity");
    setCard("aim-prediction_velocity_deadzone_px_s", predOn && method === "velocity");
    setCard("aim-aim_ema_smoothing", !!(el("aim-aim_ema_smoothing_enabled") && el("aim-aim_ema_smoothing_enabled").checked));
    // Same parent -> child rules as aim_page.py (display only; hidden options keep their values).
    function on(id) { return !!(el(id) && el(id).checked); }
    var stickyOn = on("aim-sticky_lock_enabled");
    setCard("aim-prediction_horizon_ms", predOn);
    setCard("aim-prediction_max_velocity", predOn);
    setCard("aim-prediction_history_len", predOn && method !== "anchor");
    setCard("aim-lock_persistence_ms", stickyOn);
    ["lost_target_timeout_ms", "min_lock_duration_ms", "target_switch_threshold", "reacquire_radius_px", "tracking_confidence"]
      .forEach(function (k) { setCard("aim-" + k, stickyOn); });
    setCard("aim-kalman_process_noise", on("aim-kalman_enabled"));
    setCard("aim-kalman_measurement_noise", on("aim-kalman_enabled"));
    ["aim_y_reduce_delay", "aim_y_reduce_floor", "aim_y_reduce_ramp", "aim_y_reduce_settle_px", "aim_y_vel_restore_px_s"].forEach(function (k) {
      setCard("aim-" + k, on("aim-aim_y_reduce_enabled"));
    });
    setCard("aim-target_priority_confidence_weight", !!(el("aim-target_priority_mode") && el("aim-target_priority_mode").value === "composite"));

    // Anchor Estimator: shown while Box Prediction (Inference tab, server extra anchor_enabled) or the Aim
    // Lead with the Anchor method uses it (same rule as aim_page._updateAnchorEstimatorVisibility()).
    var d = lastSettings.aim || {};
    var shared = !!d.anchor_enabled || (predOn && method === "anchor");
    var sec = el("aim-anchor-est-sec");
    if (sec) sec.classList.toggle("hidden", !shared);
    var amode = (el("aim-anchor_mode") || {}).value || d.anchor_mode || "adaptive";
    ["mode", "strength", "accel", "conf", "history", "maxdist"].forEach(function (k) {
      var c = el("aim-anchor-" + k + "-card"); if (c) c.classList.toggle("hidden", !shared);
    });
    var sm = el("aim-anchor-smoothing-card");
    if (sm) sm.classList.toggle("hidden", !(shared && (amode === "smoothed" || amode === "adaptive")));
  }

  function updateHumanizationVisibility() {
    var master = document.getElementById("aim-humanization-enabled").checked;
    function gate(toggleId, groupId) {
      var toggle = document.getElementById(toggleId);
      var enabled = master && !!(toggle && toggle.checked);
      document.getElementById(groupId).classList.toggle("hidden", !enabled);
    }
    gate("aim-humanization-micro-jitter-enabled", "aim-humanization-jitter-subgroup");
  }

  document.getElementById("aim-humanization-reset-btn").addEventListener("click", function () {
    var btn = document.getElementById("aim-humanization-reset-btn");
    btn.disabled = true;
    actionCall("Could not reset humanization", "POST", "/api/control/humanization_reset", {}, "Humanization reset to defaults")
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
  function applyPidUnsafeSliderLimits(enabled) {
    // Mirror aim_page.py: safe mode caps P travel at 0.5; Unsafe Mode opens to 1.0.
    var max = enabled ? "1" : "0.5";
    ["aim-pid_kp_x", "aim-pid_kp_y"].forEach(function (id) {
      var el = document.getElementById(id);
      if (!el) return;
      el.max = max;
      if (el._pairedRange) {
        el._pairedRange.max = max;
        if (typeof updateRangeFill === "function") updateRangeFill(el._pairedRange);
      }
    });
  }

  document.getElementById("aim-pid_unsafe_mode").addEventListener("change", function (e) {
    var enabled = e.target.checked;
    applyPidUnsafeSliderLimits(enabled);
    var body = { pid_unsafe_mode: enabled };
    if (!enabled) {
      var kpX = document.getElementById("aim-pid_kp_x");
      var kpY = document.getElementById("aim-pid_kp_y");
      var xVal = parseFloat(kpX.value);
      var yVal = parseFloat(kpY.value);
      if (!isNaN(xVal) && xVal > 0.5) { body.pid_kp_x = 0.5; kpX.value = "0.5"; }
      if (!isNaN(yVal) && yVal > 0.5) { body.pid_kp_y = 0.5; kpY.value = "0.5"; }
      if (kpX._pairedRange) kpX._pairedRange.value = kpX.value;
      if (kpY._pairedRange) kpY._pairedRange.value = kpY.value;
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

  // mouse_move_method is a plain generic field now: makcu / vigem_network also set the matching
  // click method server-side (core/aim_coupling.py via web_control_settings._post_apply), the same
  // rule the Qt Aim page uses, so there is no second POST here to keep in step. The Keys & HW and
  // Trigger panels re-fetch their own state whenever they are activated.

  // ---------------------------------------------------------------------
  // Keys & HW panel — MAKCU-mode visibility, hotkey VK-select population,
  // MAKCU COM port enumeration.
  // ---------------------------------------------------------------------
  function updateKeysVisibility(data) {
    data = data || {};
    var isMakcu = data.mouse_move_method === "makcu";
    // Live DOM wins for fields this panel owns, so a change event updates
    // visibility before the next settings poll lands.
    var hotkeysEl = document.getElementById("keys-makcu_device_hotkeys");
    var deviceHotkeys = hotkeysEl ? hotkeysEl.checked : !!data.makcu_device_hotkeys;
    var connEl = document.getElementById("keys-makcu_connection");
    var udp = (connEl ? connEl.value : data.makcu_connection) === "udp";
    var spreadEl = document.getElementById("keys-makcu_mouse_spread_enabled");
    var spreadOn = spreadEl ? spreadEl.checked : !!data.makcu_mouse_spread_enabled;
    // Aim keys 1–3 hide in MAKCU mode; the toggle key stays, matching
    // keys_page.py. Auto-fire keys come back when hotkeys are read through
    // the device (any VK is bindable again on a 2-PC setup).
    ["keys-aim-key-1-card", "keys-aim-key-2-card", "keys-aim-key-3-card"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", isMakcu);
    });
    ["keys-fire-group", "keys-fire-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", isMakcu && !deviceHotkeys);
    });
    ["keys-makcu-conn-group", "keys-makcu-conn-group-title", "keys-makcu-keys-group", "keys-makcu-keys-group-title"].forEach(function (id) {
      document.getElementById(id).classList.toggle("hidden", !isMakcu);
    });
    document.getElementById("keys-makcu-udp-card").classList.toggle("hidden", !udp);
    document.getElementById("keys-makcu-com-card").classList.toggle("hidden", udp);
    document.getElementById("keys-makcu-baud-card").classList.toggle("hidden", udp);
    document.getElementById("keys-makcu-spread-card").classList.toggle("hidden", !spreadOn);
    document.getElementById("keys-makcu-inference-card").classList.toggle("hidden", !!data.keep_detecting);
    // always_aim-gated cards are also kept live-synced from every status
    // poll (see applyStatus()) regardless of which tab is active — set
    // here too so a fresh tab activation doesn't show a stale state for
    // the instant before the next poll tick lands.
    var hideForAlwaysAim = !!data.always_aim;
    document.getElementById("keys-makcu-trigger-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-aim-mode-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-disengage-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-always-aim-button-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-always-aim-mode-card").classList.toggle("hidden", hideForAlwaysAim);
    var isVigemNet = data.mouse_move_method === "vigem_network";
    ["keys-vigem-net-group", "keys-vigem-net-group-title", "keys-remote-btn-group", "keys-remote-btn-group-title"].forEach(function (id) {
      var el = document.getElementById(id);
      if (el) el.classList.toggle("hidden", !isVigemNet);
    });
    // The remote aim button only matters once receiving is switched on (live DOM wins over the last fetch).
    var remoteEl = document.getElementById("keys-remote_btn_enabled");
    var remoteOn = remoteEl ? remoteEl.checked : !!data.remote_btn_enabled;
    var remoteCard = document.getElementById("keys-remote_aim_button");
    if (remoteCard && remoteCard.closest(".field-card")) remoteCard.closest(".field-card").classList.toggle("hidden", !remoteOn);
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

  // COM ports: the saved port is always an option (even before the list is read or if the device is
  // unplugged), then the live list is merged in. Read once when the Keys tab first loads, and by Refresh.
  var serialPortsLoaded = false;
  function refreshSerialPorts(userInitiated) {
    var btn = document.getElementById("keys-makcu-port-refresh-btn");
    if (btn) btn.disabled = true;
    return api("GET", "/api/serial_ports")
      .then(function (data) {
        if (!data || data.ok === false) throw new Error("no port list");
        serialPortsLoaded = true;
        var select = document.getElementById("keys-makcu_com_port");
        var current = select.value || ((lastSettings.keys || {}).makcu_com_port || "");
        select.innerHTML = "";
        (data.ports || []).forEach(function (port) { ensureOption(select, port); });
        ensureOption(select, current);
        if (current) select.value = current;
        if (userInitiated) notify((data.ports || []).length ? "Ports refreshed" : "No serial ports found");
      })
      .catch(function (err) {
        if (userInitiated) notify("Could not list serial ports: " + (err && err.message ? err.message : "request failed"), true);
      })
      .then(function () { if (btn) btn.disabled = false; });
  }
  document.getElementById("keys-makcu-port-refresh-btn").addEventListener("click", function () {
    refreshSerialPorts(true);
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

  // Sets a select from the host's value without fighting the operator: a focused (open) select is left
  // alone, and the option list is only rebuilt when it actually changed.
  function syncSelect(select, value, label) {
    if (!select || value === undefined || value === null || value === "") return;
    if (document.activeElement === select) return;
    ensureOption(select, value, label);
    if (select.value !== String(value)) select.value = String(value);
  }

  function applyCaptureExtras(data) {
    document.getElementById("cap-udp-system-ip").textContent = "Stream to: " + (data.system_ip || "—");

    var bindSelect = document.getElementById("cap-udp-bind-ip-select");
    var bindOptions = data.bind_ip_options || ["0.0.0.0"];
    var haveOptions = Array.prototype.map.call(bindSelect.options, function (o) { return o.value; });
    if (document.activeElement !== bindSelect && haveOptions.join("|") !== bindOptions.join("|")) {
      bindSelect.innerHTML = "";
      bindOptions.forEach(function (ip) { ensureOption(bindSelect, ip); });
    }
    syncSelect(bindSelect, data.udp_bind_ip);

    syncSelect(document.getElementById("cap-uvc-device-select"), data.uvc_device_index, "Device " + data.uvc_device_index);

    if (data.uvc_width && data.uvc_height) {
      syncSelect(document.getElementById("cap-uvc-resolution-select"), data.uvc_width + "x" + data.uvc_height);
    }

    syncSelect(document.getElementById("cap-uvc-fps-select"), data.uvc_fps);
    syncSelect(document.getElementById("cap-ndi-source-select"), data.ndi_source_name);

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

    var stateEl = document.getElementById("cap-preview-state");
    var shouldRun = isStream && currentTab === "capture" && !previewHidden;
    if (!img.dataset.wired) {
      img.dataset.wired = "1";
      // An MJPEG <img> fires load on its first frame and error if the
      // stream dies — say so instead of showing a blank/broken-image box.
      img.addEventListener("load", function () { stateEl.textContent = ""; });
      img.addEventListener("error", function () {
        if (!img.dataset.streaming) return;
        stateEl.textContent = "Preview interrupted — retrying…";
        clearTimeout(img._retryTimer);
        img._retryTimer = setTimeout(function () {
          if (img.dataset.streaming) {
            img.src = "/api/preview_stream?preview_token=" + encodeURIComponent(getToken()) + "&t=" + Date.now();
          }
        }, 3000);
      });
    }
    if (shouldRun) {
      if (!img.dataset.streaming) {
        img.dataset.streaming = "1";
        stateEl.textContent = "Waiting for the first frame from the capture source…";
        img.src = "/api/preview_stream?preview_token=" + encodeURIComponent(getToken());
      }
    } else if (img.dataset.streaming) {
      clearTimeout(img._retryTimer);
      img.removeAttribute("src");
      delete img.dataset.streaming;
      stateEl.textContent = previewHidden ? "Preview hidden." : "";
    }
  }

  function updateUvcSubVisibility() {
    var methodEl = document.getElementById("cap-uvc_capture_method");
    var isDshow = !!(methodEl && methodEl.value === "dshow");
    var engEl = document.getElementById("cap-uvc_engine");
    var engine = engEl ? engEl.value : "opencv";
    document.getElementById("cap-uvc-engine-card").classList.toggle("hidden", !isDshow);
    document.getElementById("cap-uvc-ffmpeg-path-card").classList.toggle("hidden", !(isDshow && engine === "ffmpeg"));
    // The native DLL only implements MJPEG and NV12 (as in the Qt page): grey the other formats
    // out instead of rewriting the saved choice, so leaving the native engine restores it.
    var fmtEl = document.getElementById("cap-uvc_video_format");
    if (fmtEl) {
      Array.prototype.forEach.call(fmtEl.options, function (o) {
        o.disabled = isDshow && engine === "native" && o.value !== "mjpeg" && o.value !== "nv12";
      });
    }
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
            // Desktop always persists Model_Hud/<basename> — option values
            // must match that form or select.value = config path fails and a
            // generic POST can write a bare basename back into Config.
            opt.value = "Model_Hud/" + name;
            opt.textContent = name;
            select.appendChild(opt);
          });
          if (lastSettings.model && lastSettings.model.hud_model_path) {
            var want = lastSettings.model.hud_model_path.replace(/\\/g, "/");
            if (want.indexOf("/") < 0) want = "Model_Hud/" + want;
            select.value = want;
          }
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

  // Corner Bracket Length only applies while boxes are drawn as brackets.
  // Each line-thickness card is shown only while its Show toggle is on (mirrors visuals_page.py).
  var LINE_THICKNESS_CARDS = [
    ["vis-show_fov", "vis-fov-thickness-card"],
    ["vis-show_detect_range", "vis-range-thickness-card"],
    ["vis-show_tracer_line", "vis-tracer-thickness-card"],
    ["vis-show_tracer_line", "vis-tracer-origin-card"],
    ["vis-show_aim_prediction_marker", "vis-marker-thickness-card"],
  ];
  var LINE_THICKNESS_TOGGLE_KEYS = ["show_fov", "show_detect_range", "show_tracer_line", "show_aim_prediction_marker"];
  function updateLineThicknessVisibility() {
    var predOn = !!((lastSettings.visuals || {}).prediction_or_kalman);
    LINE_THICKNESS_CARDS.forEach(function (pair) {
      var toggle = document.getElementById(pair[0]);
      var card = document.getElementById(pair[1]);
      var show = !!(toggle && toggle.checked);
      if (pair[1] === "vis-marker-thickness-card") show = show && predOn;
      if (card) card.classList.toggle("hidden", !show);
    });
  }

  // Show an option only while what it draws / configures can exist (same rules as visuals_page.py). Display
  // only: hidden options keep their values.
  function updateVisualsContextVisibility(data) {
    data = data || lastSettings.visuals || {};
    function setHidden(id, hide) {
      var el = document.getElementById(id);
      if (el) el.classList.toggle("hidden", !!hide);
    }
    setHidden("vis-stick-gauge-card", data.mouse_move_method !== "vigem_network");
    setHidden("vis-anchor-debug-card", !data.anchor_enabled);
    setHidden("vis-aim-marker-card", !data.prediction_or_kalman);
    var acrylic = document.getElementById("vis-enable_acrylic");
    setHidden("vis-acrylic-alpha-card", !(acrylic && acrylic.checked));
    var webOn = !!(document.getElementById("vis-web_esp_enabled") || {}).checked;
    ["vis-web-esp-http-card", "vis-web-esp-ws-card", "vis-web-esp-fps-card", "vis-web-esp-connect-card"].forEach(function (id) {
      setHidden(id, !webOn);
    });
    updateLineThicknessVisibility();
  }

  function updateBoxStyleVisibility() {
    var full = document.getElementById("vis-box_full_rect");
    var card = document.getElementById("vis-box-corner-card");
    if (card) card.classList.toggle("hidden", !!(full && full.checked));
  }

  // GPU sensor switches and refresh only matter while GPU Monitoring is on.
  function updateGpuMonitorVisibility() {
    var master = document.getElementById("vis-status_panel_show_gpu");
    var on = !!(master && master.checked);
    document.querySelectorAll(".gpu-sensor-card").forEach(function (card) {
      card.classList.toggle("hidden", !on);
    });
  }

  function applyVisualsExtras(data) {
    var toggle = document.getElementById("vis-web_esp_enabled");
    if (toggle !== document.activeElement) {
      suppressWebEspToggleEcho = true;
      toggle.checked = !!data.web_esp_enabled;
      suppressWebEspToggleEcho = false;
    }
    var urlEl = document.getElementById("vis-web-esp-url");
    urlEl.textContent = data.web_esp_running && data.web_esp_url ? data.web_esp_url : "—  (not running)";
    var gpuStatus = document.getElementById("vis-gpu-monitor-status");
    if (gpuStatus) gpuStatus.textContent = data.gpu_monitor_status || "";
  }

  function scheduleWebEspRestart() {
    if (webEspRestartTimer) clearTimeout(webEspRestartTimer);
    webEspRestartTimer = setTimeout(function () {
      webEspRestartTimer = null;
      actionCall("Could not restart Web ESP", "POST", "/api/control/web_esp_restart", {})
        .then(function () { loadTabSettings("visuals", true); });
    }, 600);
  }

  document.getElementById("vis-web_esp_enabled").addEventListener("change", function (e) {
    if (suppressWebEspToggleEcho) return;
    var enabled = e.target.checked;
    e.target.disabled = true;
    actionCall("Could not " + (enabled ? "start" : "stop") + " Web ESP", "POST", "/api/control/web_esp_enabled", { enabled: enabled })
      .then(function (data) {
        if (!data) e.target.checked = !enabled;
        else loadTabSettings("visuals", true);
        updateVisualsContextVisibility();
      })
      .then(function () { e.target.disabled = false; });
  });

  // Opens on the HOST machine running Axiom, not this remote browser tab
  // — same caveat as Model panel's Open Model Folder button.
  document.getElementById("vis-web-esp-open-btn").addEventListener("click", function () {
    var btn = document.getElementById("vis-web-esp-open-btn");
    btn.disabled = true;
    actionCall("Could not open Web ESP", "POST", "/api/control/web_esp_open", {}, "Opened on the Axiom PC")
      .then(function () { btn.disabled = false; });
  });

  // ---------------------------------------------------------------------
  // Configs panel — TWO independent CRUD boxes sharing one generic wiring
  // function, wireConfigCrudPanel(): a "Config" box (full settings
  // snapshots, /api/full_configs/*) and a "Preset" box (aim-only,
  // /api/configs/*, unchanged from before) — mirroring configs_page.py's
  // own two-_ManagerBox split in the Qt app. Plus content-based
  // Export/Import since a browser can't supply a host-side file path (see
  // web_control_settings.py's export_config_preset_content()/
  // import_config_preset_content() docstrings). window.prompt()/confirm()
  // stand in for configs_page.py's QInputDialog/QMessageBox — the direct
  // browser-native equivalent for this plain, no-framework client.
  // ---------------------------------------------------------------------
  // Card list over a config/preset <select size="8">: one row per entry
  // (name, "Loaded" pill), so entries read as separate items.
  // The <select> stays the source of truth — every button reads .value.
  function enhanceConfigList(selectEl, noun, loadBtn) {
    var list = document.createElement("div");
    list.className = "cfg-list";
    list.setAttribute("role", "listbox");
    list.setAttribute("tabindex", "0");
    list.setAttribute("aria-label", noun.charAt(0).toUpperCase() + noun.slice(1) + "s");
    selectEl.classList.add("cfg-native");
    selectEl.setAttribute("tabindex", "-1");
    selectEl.setAttribute("aria-hidden", "true");
    selectEl.parentNode.insertBefore(list, selectEl.nextSibling);
    var activeName = "";

    function choose(name) {
      if (selectEl.value === name) return;
      selectEl.value = name;
      selectEl.dispatchEvent(new Event("change", { bubbles: true }));
      render();
    }
    function render() {
      list.innerHTML = "";
      var opts = Array.prototype.slice.call(selectEl.options);
      if (!opts.length) {
        var empty = document.createElement("div");
        empty.className = "cfg-empty";
        empty.textContent = noun === "preset"
          ? "No presets yet — Create saves the current aim settings."
          : "No configs yet — Create saves every current setting.";
        list.appendChild(empty);
        list.removeAttribute("aria-activedescendant");
        return;
      }
      opts.forEach(function (o, i) {
        var row = document.createElement("div");
        row.className = "cfg-row";
        row.id = selectEl.id + "-row-" + i;
        row.setAttribute("role", "option");
        var selected = o.value === selectEl.value && selectEl.selectedIndex >= 0;
        row.setAttribute("aria-selected", selected ? "true" : "false");
        if (selected) { row.classList.add("selected"); list.setAttribute("aria-activedescendant", row.id); }
        var name = document.createElement("span");
        name.className = "cfg-name";
        name.textContent = o.value;
        row.appendChild(name);
        if (activeName && o.value === activeName) {
          row.classList.add("active");
          var pill = document.createElement("span");
          pill.className = "cfg-pill";
          pill.textContent = "Loaded";
          row.appendChild(pill);
        }
        row.addEventListener("click", function () { choose(o.value); });
        row.addEventListener("dblclick", function () { choose(o.value); loadBtn.click(); });
        list.appendChild(row);
      });
      var sel = list.querySelector(".cfg-row.selected");
      if (sel && sel.scrollIntoView) sel.scrollIntoView({ block: "nearest" });
    }
    list.addEventListener("keydown", function (e) {
      var opts = selectEl.options;
      if (!opts.length) return;
      var i = selectEl.selectedIndex;
      if (e.key === "ArrowDown") i = Math.min(opts.length - 1, i + 1);
      else if (e.key === "ArrowUp") i = Math.max(0, i < 0 ? 0 : i - 1);
      else if (e.key === "Home") i = 0;
      else if (e.key === "End") i = opts.length - 1;
      else if (e.key === "Enter") { if (i >= 0) loadBtn.click(); e.preventDefault(); return; }
      else return;
      e.preventDefault();
      choose(opts[i].value);
    });
    return {
      render: render,
      setActive: function (name) {
        name = name || "";
        if (name === activeName) return;
        activeName = name;
        render();
      },
    };
  }

  function wireConfigCrudPanel(cfg) {
    var selectEl = document.getElementById(cfg.selectId);
    var reasonEl = document.getElementById(cfg.reasonId);
    var cardList = enhanceConfigList(selectEl, cfg.noun, document.getElementById(cfg.loadBtnId));

    function refreshList() {
      fetch(cfg.apiBase, { headers: authHeaders() })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (!data || !data.presets) return;
          var current = selectEl.value;
          selectEl.innerHTML = "";
          data.presets.forEach(function (name) {
            var opt = document.createElement("option");
            opt.value = name;
            opt.textContent = name;
            selectEl.appendChild(opt);
          });
          var has = Array.prototype.some.call(selectEl.options, function (o) { return o.value === current; });
          if (has) selectEl.value = current;
          else selectEl.selectedIndex = -1;
          cardList.render();
          if (cfg.onListRefreshed) cfg.onListRefreshed(data.presets);
        })
        .catch(function () {});
    }

    document.getElementById(cfg.refreshBtnId).addEventListener("click", refreshList);

    document.getElementById(cfg.createBtnId).addEventListener("click", function () {
      var name = window.prompt("New " + cfg.noun + " name:");
      if (!name) return;
      fetch(cfg.apiBase + "/save", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ name: name }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (!data || data.ok === false) {
            reasonEl.textContent = (data && data.reason) || "failed to create";
          } else {
            reasonEl.textContent = "created \"" + name + "\"";
            refreshList();
          }
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    document.getElementById(cfg.saveBtnId).addEventListener("click", function () {
      var name = selectEl.value;
      if (!name) { reasonEl.textContent = "select a " + cfg.noun + " first"; return; }
      if (!window.confirm('Overwrite "' + name + '" with the current live settings?')) return;
      fetch(cfg.apiBase + "/save", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ name: name }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          reasonEl.textContent = (data && data.ok) ? "saved \"" + name + "\"" : ((data && data.reason) || "failed to save");
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    document.getElementById(cfg.loadBtnId).addEventListener("click", function () {
      var name = selectEl.value;
      if (!name) { reasonEl.textContent = "select a " + cfg.noun + " first"; return; }
      // Dry-run preview first (mirrors configs_page.py's Load button — see
      // ConfigManager.preview_config_changes()) so the operator sees what
      // would actually change before it's applied, instead of silently
      // overwriting the live config.
      fetch(cfg.apiBase + "/preview?name=" + encodeURIComponent(name), { headers: authHeaders() })
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
          fetch(cfg.apiBase + "/load", {
            method: "POST",
            headers: authHeaders(),
            body: JSON.stringify({ name: name }),
          })
            .then(function (res) { return res.ok ? res.json() : null; })
            .then(function (loadData) {
              if (loadData && loadData.ok) {
                reasonEl.textContent = "loaded \"" + name + "\"";
                // The just-loaded preset/config can touch fields on any
                // tab — resync every generic-schema tab so the whole
                // client reflects the new live config, not just Configs.
                resyncAllTabs();
              } else {
                reasonEl.textContent = (loadData && loadData.reason) || "failed to load";
              }
            })
            .catch(function () { reasonEl.textContent = "request failed"; });
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    document.getElementById(cfg.renameBtnId).addEventListener("click", function () {
      var oldName = selectEl.value;
      if (!oldName) { reasonEl.textContent = "select a " + cfg.noun + " first"; return; }
      var newName = window.prompt("Rename \"" + oldName + "\" to:", oldName);
      if (!newName || newName === oldName) return;
      fetch(cfg.apiBase + "/rename", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ old_name: oldName, new_name: newName }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (data && data.ok) {
            reasonEl.textContent = "renamed to \"" + newName + "\"";
            refreshList();
          } else {
            reasonEl.textContent = (data && data.reason) || "failed to rename";
          }
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    document.getElementById(cfg.deleteBtnId).addEventListener("click", function () {
      var name = selectEl.value;
      if (!name) { reasonEl.textContent = "select a " + cfg.noun + " first"; return; }
      if (!window.confirm('Delete "' + name + '"? This cannot be undone.')) return;
      fetch(cfg.apiBase + "/delete", {
        method: "POST",
        headers: authHeaders(),
        body: JSON.stringify({ name: name }),
      })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (data && data.ok) {
            reasonEl.textContent = "deleted \"" + name + "\"";
            refreshList();
          } else {
            reasonEl.textContent = (data && data.reason) || "failed to delete";
          }
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    // Export downloads the raw JSON via a Blob + temporary <a download>
    // link — this page is the app's own hosted static client, not a
    // sandboxed Artifact, so a download link works normally here.
    document.getElementById(cfg.exportBtnId).addEventListener("click", function () {
      var name = selectEl.value;
      if (!name) { reasonEl.textContent = "select a " + cfg.noun + " first"; return; }
      fetch(cfg.apiBase + "/export?name=" + encodeURIComponent(name), { headers: authHeaders() })
        .then(function (res) { return res.ok ? res.json() : null; })
        .then(function (data) {
          if (!data || data.ok === false) {
            reasonEl.textContent = (data && data.reason) || "failed to export";
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
          reasonEl.textContent = "exported \"" + name + "\"";
        })
        .catch(function () { reasonEl.textContent = "request failed"; });
    });

    // Import reads the chosen file's content client-side (FileReader) and
    // POSTs it as JSON text — there's no host-side path a browser could
    // send instead, see import_config_preset_content()'s docstring.
    var importFileEl = document.getElementById(cfg.importFileId);
    document.getElementById(cfg.importBtnId).addEventListener("click", function () {
      importFileEl.click();
    });

    importFileEl.addEventListener("change", function () {
      var file = importFileEl.files && importFileEl.files[0];
      importFileEl.value = ""; // allow re-selecting the same file later
      if (!file) return;
      var reader = new FileReader();
      reader.onload = function () {
        fetch(cfg.apiBase + "/import", {
          method: "POST",
          headers: authHeaders(),
          body: JSON.stringify({ content: reader.result }),
        })
          .then(function (res) { return res.ok ? res.json() : null; })
          .then(function (data) {
            if (data && data.ok) {
              reasonEl.textContent = "imported as \"" + data.name + "\"";
              refreshList();
            } else {
              reasonEl.textContent = (data && data.reason) || "failed to import";
            }
          })
          .catch(function () { reasonEl.textContent = "request failed"; });
      };
      reader.onerror = function () { reasonEl.textContent = "failed to read file"; };
      reader.readAsText(file);
    });

    // Opens on the HOST machine running Axiom, not this remote browser tab
    // — same caveat as Model panel's Open Model Folder button.
    document.getElementById(cfg.openFolderBtnId).addEventListener("click", function () {
      var btn = document.getElementById(cfg.openFolderBtnId);
      btn.disabled = true;
      actionCall("Could not open the folder", "POST", cfg.openFolderRoute, {}, "Opened on the Axiom PC")
        .then(function () { btn.disabled = false; });
    });

    cardList.render();
    return { refreshList: refreshList, setActive: cardList.setActive };
  }

  var _presetPanel = wireConfigCrudPanel({
    apiBase: "/api/configs",
    noun: "preset",
    selectId: "configs-select",
    reasonId: "configs-reason",
    refreshBtnId: "configs-refresh-btn",
    createBtnId: "configs-create-btn",
    saveBtnId: "configs-save-btn",
    loadBtnId: "configs-load-btn",
    renameBtnId: "configs-rename-btn",
    deleteBtnId: "configs-delete-btn",
    exportBtnId: "configs-export-btn",
    importBtnId: "configs-import-btn",
    importFileId: "configs-import-file",
    openFolderBtnId: "configs-open-folder-btn",
    openFolderRoute: "/api/control/open_configs_folder",
    // The Quick Presets sidebar section reuses this exact fetch rather
    // than making its own — one master list, same "allModelNames"
    // precedent the Model/Convert tabs already share.
    onListRefreshed: function (names) { populatePresetSlotSelects(names); },
  });
  function refreshConfigsList() { _presetPanel.refreshList(); }

  var _fullConfigPanel = wireConfigCrudPanel({
    apiBase: "/api/full_configs",
    noun: "config",
    selectId: "full-configs-select",
    reasonId: "full-configs-reason",
    refreshBtnId: "full-configs-refresh-btn",
    createBtnId: "full-configs-create-btn",
    saveBtnId: "full-configs-save-btn",
    loadBtnId: "full-configs-load-btn",
    renameBtnId: "full-configs-rename-btn",
    deleteBtnId: "full-configs-delete-btn",
    exportBtnId: "full-configs-export-btn",
    importBtnId: "full-configs-import-btn",
    importFileId: "full-configs-import-file",
    openFolderBtnId: "full-configs-open-folder-btn",
    openFolderRoute: "/api/control/open_full_configs_folder",
  });
  function refreshFullConfigsList() { _fullConfigPanel.refreshList(); }

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

  // The server's (or cache's) assigned name per slot. The options list and
  // the assignments arrive from two independent requests in either order; if
  // the assignment lands first, `sel.value = name` has no matching <option>
  // yet and is a silent no-op, and the later option rebuild would then reset
  // the slot to "unassigned" while its Load button stayed enabled. Keeping
  // the assignment here lets populatePresetSlotSelects() re-apply it.
  var presetSlotAssigned = ["", "", "", "", ""];

  function populatePresetSlotSelects(names) {
    // Cached so the next page load can rebuild these <option> lists
    // immediately, before GET /api/configs has even round-tripped — see
    // the restore call near the bottom of this file's startup block.
    setLocalJSON(PRESET_NAMES_CACHE_KEY, names || []);
    presetSlotSelects.forEach(function (sel, i) {
      if (document.activeElement === sel) return; // don't clobber a mid-pick
      var current = presetSlotAssigned[i] || sel.value;
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

  // Reads the 5 selects' current values back out, for caching after any
  // successful change — a plain array is easier to restore from than
  // re-deriving it, and this only ever runs after a real assignment
  // succeeds, so it always reflects genuine server-confirmed state.
  function currentPresetSlotValues() {
    return presetSlotSelects.map(function (sel) { return sel ? sel.value : ""; });
  }

  function fetchPresetSlotAssignments() {
    fetch("/api/preset_slots", { headers: authHeaders() })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || !data.slots) return;
        data.slots.forEach(function (name, i) {
          var sel = presetSlotSelects[i];
          if (!sel) return;
          presetSlotAssigned[i] = name || "";
          if (document.activeElement === sel) return;
          sel.value = name || "";
          presetSlotLoadBtns[i].disabled = !name;
        });
        setLocalJSON(PRESET_SLOTS_CACHE_KEY, data.slots);
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
            presetSlotAssigned[i] = name || "";
            presetSlotLoadBtns[i].disabled = !name;
            setLocalJSON(PRESET_SLOTS_CACHE_KEY, currentPresetSlotValues());
            notify(name ? "Slot " + (i + 1) + " → " + name : "Slot " + (i + 1) + " cleared");
          } else {
            notify("Could not assign slot " + (i + 1), true);
            // Refused (e.g. the picked name no longer exists) — resync
            // this row from the server's actual current assignment rather
            // than trusting the now-stale local selection.
            fetchPresetSlotAssignments();
          }
        })
        .catch(function () { notify("Could not assign slot " + (i + 1) + ": cannot reach the Axiom host", true); });
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
            resyncAllTabs();
            notify("Loaded preset: " + name);
          } else {
            notify("Could not load preset " + name + (data && data.reason ? " (" + String(data.reason).replace(/_/g, " ") + ")" : ""), true);
          }
        })
        .catch(function () { notify("Could not load preset " + name + ": cannot reach the Axiom host", true); })
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
  // (also never written back to Config). Precision is an FP32/FP16 combo.
  // The last-used value still comes from trt_fp16_enabled (mirroring
  // ConvertPage.setConfig()) but is not written on a bare change; the real
  // write happens only after a successful build, exactly when
  // _onConvertFinished() does it.
  // ---------------------------------------------------------------------
  var convertModelSelect = document.getElementById("convert-model-select");
  var convertPrecisionSelect = document.getElementById("convert-precision-select");
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
          // A successful build just created a new cached .engine — refresh
          // so the Model tab's ✓/⬇ badges pick it up (whether this build
          // was started from here, the Qt GUI, or auto-redirected via
          // needs_conversion) instead of showing stale ⬇ until next reload.
          if (data.success) loadModelList();
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
    if (fp16 === undefined) fp16 = !convertPrecisionSelect || convertPrecisionSelect.value !== "fp32";
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

  var lastModelLoadError = "";
  function applyModelLoadError(err) {
    // Hot-swap load failures (ai_loop keeps the old model) — show them in the
    // Model panel's hint line; clear only our own message once it resolves.
    err = err || "";
    if (err === lastModelLoadError) return;
    var wasShowing = modelSwitchReason && modelSwitchReason.textContent === "Model failed to load: " + lastModelLoadError;
    lastModelLoadError = err;
    if (err) setModelListHint("Model failed to load: " + err, true);
    else if (wasShowing) setModelListHint("", false);
  }

  // GPU tiles — same sensors/labels/units as the Qt Status Panel's GPU section; the server
  // sends preformatted text (core/gpu_monitor.format_reading) so no unit logic lives here.
  var GPU_TILE_KEYS = ["temp", "util", "vram", "fan", "core_clock", "mem_clock", "power"];
  var GPU_STATUS_TEXT = { checking: "Checking\u2026", no_nvml: "NVML not installed", no_gpu: "No NVIDIA GPU", not_started: "Checking\u2026" };
  function applyGpu(g) {
    var section = document.getElementById("s-gpu-section");
    if (!section) return;
    if (!g) { section.classList.add("hidden"); return; }
    section.classList.remove("hidden");
    var title = document.getElementById("s-gpu-title");
    title.textContent = g.short_name ? "GPU \u00b7 " + g.short_name : "GPU";
    title.title = g.name || "";
    var ok = g.status === "ok";
    var msgStat = document.getElementById("s-gpu-msg-stat");
    msgStat.classList.toggle("hidden", ok);
    if (!ok) document.getElementById("s-gpu-msg").textContent = GPU_STATUS_TEXT[g.status] || "\u2014";
    var supported = g.supported || [];
    GPU_TILE_KEYS.forEach(function (key) {
      var stat = document.getElementById("s-gpu-" + key + "-stat");
      var shown = ok && supported.indexOf(key) !== -1 && g.text && (key in g.text);
      stat.classList.toggle("hidden", !shown);
      if (shown) document.getElementById("s-gpu-" + key).textContent = g.text[key];
    });
  }

  // "Show advanced settings" (sidebar switch, same persisted flag as the Qt app). initAdvancedSettings()
  // installs the setter; applyStatus() feeds it the server's value on every poll.
  var advancedOn = true;
  var advancedPending = false;
  var setAdvancedFromServer = null;

  function applyStatus(s) {
    if (setAdvancedFromServer && typeof s.show_advanced_settings === "boolean" && !advancedPending) {
      setAdvancedFromServer(s.show_advanced_settings);
    }
    applyModelLoadError(s.model_load_error);
    noteModelRuntime(s);
    applyGpu(s.gpu);
    document.getElementById("s-running").textContent = s.running ? "RUNNING" : "stopped";
    document.getElementById("s-active").textContent = s.active ? "ON" : "OFF";
    document.getElementById("s-firing").textContent = s.aim_firing ? "Aiming" : "Idle";
    document.getElementById("s-model").textContent = s.model || "—";
    _presetPanel.setActive(s.active_preset_name);
    _fullConfigPanel.setActive(s.active_config_name);
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
    document.getElementById("keys-makcu-always-aim-button-card").classList.toggle("hidden", hideForAlwaysAim);
    document.getElementById("keys-makcu-always-aim-mode-card").classList.toggle("hidden", hideForAlwaysAim);

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
    if (vigemNetToggle && !vigemNetToggle.disabled) {
      suppressVigemNetToggleEcho = true;
      vigemNetToggle.checked = !!s.vigem_network_connected;
      suppressVigemNetToggleEcho = false;
    }

    // A staged pick that now equals the host's value is simply applied; otherwise it is left alone.
    if (modelStaged.model && s.model && modelSelect.value === s.model) modelStaged.model = false;
    if (modelStaged.backend && s.selected_backend && backendSelect.value === displayBackend(s.selected_backend)) modelStaged.backend = false;
    if (modelStaged.precision && typeof s.trt_fp16_enabled === "boolean") {
      var pSel = document.getElementById("model-precision-select");
      if (pSel && pSel.value === (s.trt_fp16_enabled ? "fp16" : "fp32")) modelStaged.precision = false;
    }
    updateModelPendingHint();
    if (s.model && !modelStaged.model && !modelSwitchPending && document.activeElement !== modelSelect) {
      applyModelSelection(s.model);                 // also follows a model change made in the Qt app
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
    var modelPrecisionSelect = document.getElementById("model-precision-select");
    if (modelPrecisionSelect && typeof s.trt_fp16_enabled === "boolean"
        && document.activeElement !== modelPrecisionSelect && !modelStaged.precision && !modelSwitchPending) {
      var precisionValue = s.trt_fp16_enabled ? "fp16" : "fp32";
      if (modelPrecisionSelect.value !== precisionValue) {
        modelPrecisionSelect.value = precisionValue;
        refreshModelOptionLabels();
      }
    }
    if (s.selected_backend && document.activeElement !== backendSelect && !modelStaged.backend && !modelSwitchPending) {
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

  // Polling: chained setTimeout (not setInterval) so a slow response can't
  // overlap the next tick, backoff while the host is unreachable, a slower
  // retry while no/bad token, and fully paused while the tab is hidden.
  var pollTimer = null;
  var pollInFlight = false;
  var lastSettingsSyncAt = 0;
  var SETTINGS_SYNC_MS = 3000;
  var POLL_MAX_BACKOFF_MS = 10000;

  function nextPollDelay() {
    if (!getToken()) return 3000;
    if (authState === "bad") return 5000;
    if (pollFailures > 0) return Math.min(POLL_MS * Math.pow(2, pollFailures - 1), POLL_MAX_BACKOFF_MS);
    return POLL_MS;
  }

  function schedulePoll() {
    clearTimeout(pollTimer);
    if (document.hidden) return; // resumed by the visibilitychange handler
    pollTimer = setTimeout(poll, nextPollDelay());
  }

  function poll() {
    clearTimeout(pollTimer);
    if (document.hidden || pollInFlight) return;

    if (!getToken()) {
      setStatusBadge(false, "enter token", "Paste the access token in the sidebar, then Save", true);
      tokenInput.classList.add("attn");
      schedulePoll();
      return;
    }
    tokenInput.classList.remove("attn");

    pollInFlight = true;
    fetch("/api/status", { headers: authHeaders() })
      .then(function (res) {
        if (res.status === 401) {
          authState = "bad";
          pollFailures = 0;
          setStatusBadge(false, "bad token", "The server rejected the access token");
          return null;
        }
        if (!res.ok) {
          pollFailures++;
          setStatusBadge(false, "server error (HTTP " + res.status + ")", "GET /api/status returned HTTP " + res.status);
          return null;
        }
        return res.json();
      })
      .then(function (data) {
        if (data) {
          authState = "ok";
          pollFailures = 0;
          setStatusBadge(true);
          applyStatus(data);

          // Keep whichever generic-schema tab is on-screen synced to the
          // live Config (a Qt-side change surfaces within a few seconds, no
          // reload). loadTabSettings()'s own guards (skip a focused field,
          // skip a field whose paired range slider is mid-drag) keep this
          // from fighting an in-progress edit here.
          var now = Date.now();
          if (GENERIC_SCHEMA_TABS.indexOf(currentTab) !== -1 && now - lastSettingsSyncAt >= SETTINGS_SYNC_MS) {
            lastSettingsSyncAt = now;
            loadTabSettings(currentTab, true);
          }
        }
      })
      .catch(function () {
        pollFailures++;
        setStatusBadge(false, "disconnected", "Cannot reach the Axiom host — retrying");
      })
      .then(function () {
        pollInFlight = false;
        schedulePoll();
      });
  }

  document.addEventListener("visibilitychange", function () {
    if (!document.hidden) {
      pollFailures = 0;
      lastSettingsSyncAt = 0;
      poll();
    }
  });

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

  if (vigemNetToggle) {
    vigemNetToggle.addEventListener("change", function () {
      if (suppressVigemNetToggleEcho) return;
      var connecting = vigemNetToggle.checked;
      var path = connecting ? "/api/control/vigem_network_connect" : "/api/control/vigem_network_disconnect";
      vigemNetToggle.disabled = true;
      fetch(path, { method: "POST", headers: authHeaders() })
        .then(function (res) {
          if (!res.ok) { vigemNetToggle.checked = !connecting; return; }
          return res.json();
        })
        .then(function (data) {
          if (data && data.ok === false) vigemNetToggle.checked = false;
        })
        .catch(function () { vigemNetToggle.checked = !connecting; })
        .then(function () { vigemNetToggle.disabled = false; });
    });
  }

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
  // The switch flow resolved (applied, refused, declined or failed): release the in-flight guard and let
  // every select resync from the host.
  function endModelSwitch() {
    modelSwitchPending = false;
    clearModelStaged();
  }

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
      endModelSwitch();
      return;
    }
    modelSwitchReason.textContent = "restarting Axiom…";
    fetch("/api/control/model_restart", {
      method: "POST",
      headers: authHeaders(),
      body: JSON.stringify({
        model_path: modelPath,
        inference_backend: inferenceBackend,
        fp16: selectedModelFp16(),
      }),
    })
      .then(function (res) { return res.ok ? res.json() : null; })
      .then(function (data) {
        if (!data || data.ok === false) {
          modelSwitchReason.textContent = describeModelSwitchReason(data && data.reason);
          // Nothing was actually applied — safe (and correct) to let the
          // next poll resync backendSelect from the still-unchanged config.
          endModelSwitch();
          return;
        }
        modelSwitchReason.textContent = "restarting Axiom — this page reconnects automatically once it's back up";
        // config.model_path/inference_backend are already written
        // server-side at this point (confirm_model_change_with_restart()
        // applies them before scheduling the restart) — releasing the
        // guard now is what lets applyStatus() start reflecting the real,
        // just-confirmed backend instead of fighting it for the remaining
        // few seconds until the process actually restarts.
        endModelSwitch();
      })
      .catch(function () {
        // The process may already be mid-restart by the time this settles
        // (the response itself can race the exit) — poll()'s own retry
        // loop silently recovers once the new process's server responds
        // again, so there's nothing more to surface here than what's
        // already showing.
        endModelSwitch();
      });
  }

  var modelPrecisionSelect = document.getElementById("model-precision-select");
  if (modelPrecisionSelect) {
    modelPrecisionSelect.addEventListener("change", function () {
      stageModelPick("precision");
      if (convertPrecisionSelect) convertPrecisionSelect.value = modelPrecisionSelect.value;
      refreshModelOptionLabels();
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
      body: JSON.stringify({
        model_path: modelPath,
        inference_backend: backendSelect.value,
        fp16: selectedModelFp16(),
      }),
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
          endModelSwitch();
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
            if (convertPrecisionSelect) convertPrecisionSelect.value = selectedModelFp16() ? "fp16" : "fp32";
            startConversionFlow(bareName, selectedModelFp16());
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
          // Which backend is effective just changed — refetch so the ✓/⬇
          // badges and legend reflect it immediately rather than the
          // pre-switch backend's cache status (or lack of badges at all).
          loadModelList();
        }
        endModelSwitch();
      })
      .catch(function (err) {
        // A thrown exception here means the request never got a response
        // at all (network error, CORS, server unreachable) — distinct
        // from the !res.ok branch above, which did get a response.
        modelSwitchReason.textContent = "request failed (" + (err && err.message ? err.message : "network error") + ")";
        endModelSwitch();
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
  updateGpuMonitorVisibility();

  loadModelList();
  // Model is the default-active panel (no click to trigger activateTab's
  // own load), so its tab settings + HUD extras are fetched explicitly here.
  loadTabSettings("model");
  ensureModelPanelExtras();

  // Seed the Quick Presets sidebar from its last-known cached state before
  // the real fetches below even land, so a page reload shows the same
  // options/selections instantly instead of a brief flash back to "—
  // unassigned —" while waiting on the network — the cache is just a fast
  // local mirror of server state, never the source of truth itself; the
  // real refreshConfigsList()/fetchPresetSlotAssignments() calls right
  // after this immediately re-confirm (and correct, if anything changed
  // server-side since the last visit) whatever this seeds.
  var cachedPresetNames = getLocalJSON(PRESET_NAMES_CACHE_KEY);
  if (cachedPresetNames) populatePresetSlotSelects(cachedPresetNames);
  var cachedPresetSlots = getLocalJSON(PRESET_SLOTS_CACHE_KEY);
  if (cachedPresetSlots) {
    cachedPresetSlots.forEach(function (name, i) {
      var sel = presetSlotSelects[i];
      if (!sel) return;
      presetSlotAssigned[i] = name || "";
      sel.value = name || "";
      if (presetSlotLoadBtns[i]) presetSlotLoadBtns[i].disabled = !name;
    });
  }

  // Quick Presets lives in the sidebar, visible regardless of which tab is
  // open — fetch its preset list + slot assignments once at startup too,
  // not gated on the Configs tab ever being activated.
  refreshConfigsList();
  fetchPresetSlotAssignments();

  // Reopen on whichever tab was last active, rather than always starting
  // back on Model — activateTab() itself handles the visibility toggle and
  // fires that tab's own settings fetch; a missing/unrecognized/"model"
  // cached value is simply left alone, since Model's own fetches just ran
  // above.
  var lastTab = getLocalJSON(LAST_TAB_KEY);
  var validTabNames = [];
  tabButtons.forEach(function (btn) { validTabNames.push(btn.getAttribute("data-tab")); });
  if (lastTab && lastTab !== "model" && validTabNames.indexOf(lastTab) !== -1) {
    activateTab(lastTab);
  }

  // ---------------------------------------------------------------------
  // UI enhancements — accessibility wiring, collapsible groups, settings
  // search, phone sidebar toggle, and the client-only Other tab. All of it
  // is progressive: it only decorates markup that already works without it.
  // ---------------------------------------------------------------------
  // ---------------------------------------------------------------------
  // Custom dropdowns. A native <select>'s popup can't be styled at all, so
  // every single-choice <select> is progressively enhanced with a styled
  // trigger + listbox menu. The real <select> stays the source of truth
  // AND the focus target (visually hidden, over the trigger): programmatic
  // .value writes, change events, disabled state and every existing
  // `document.activeElement === select` "don't clobber an edit" guard keep
  // working untouched, and screen readers still get the native control.
  // ---------------------------------------------------------------------
  function enhanceSelects() {
    // Bail out (keep native selects) unless the matching stylesheet is the
    // one that loaded — see the --axiom-css sentinel in styles.css.
    var cssVer = getComputedStyle(document.documentElement).getPropertyValue("--axiom-css").trim();
    if (cssVer !== "22") return;
    var proto = HTMLSelectElement.prototype;
    var valueDesc = Object.getOwnPropertyDescriptor(proto, "value");
    var indexDesc = Object.getOwnPropertyDescriptor(proto, "selectedIndex");
    var openDD = null;

    function closeOpen(refocus) {
      if (openDD) openDD.close(refocus);
    }
    document.addEventListener("pointerdown", function (e) {
      if (openDD && !openDD.root.contains(e.target) && !openDD.menu.contains(e.target)) closeOpen(false);
    });
    window.addEventListener("resize", function () { closeOpen(false); });
    window.addEventListener("scroll", function (e) {
      if (openDD && !openDD.menu.contains(e.target)) closeOpen(false);
    }, true);

    Array.prototype.forEach.call(document.querySelectorAll("select"), function (sel) {
      if (sel.multiple || sel.size > 1 || sel._dd) return;

      var root = document.createElement("div");
      root.className = "dd";
      var trigger = document.createElement("div");
      trigger.className = "dd-trigger";
      trigger.setAttribute("aria-hidden", "true");
      var label = document.createElement("span");
      label.className = "dd-label";
      var chev = document.createElement("span");
      chev.className = "dd-chevron";
      trigger.appendChild(label);
      trigger.appendChild(chev);

      var menu = document.createElement("div");
      menu.className = "dd-menu";
      menu.setAttribute("aria-hidden", "true");

      sel.parentNode.insertBefore(root, sel);
      root.appendChild(sel);
      root.appendChild(trigger);
      sel.classList.add("dd-native");

      var items = [];     // [{value, el}] selectable rows currently in the menu
      var active = -1;

      function refresh() {
        var opt = sel.options[sel.selectedIndex];
        label.textContent = opt ? opt.text : "—";
        label.classList.toggle("placeholder", !opt || opt.value === "");
        root.classList.toggle("disabled", sel.disabled);
      }

      function setActive(i, scroll) {
        if (!items.length) return;
        active = Math.max(0, Math.min(items.length - 1, i));
        items.forEach(function (it, k) { it.el.classList.toggle("active", k === active); });
        if (scroll) items[active].el.scrollIntoView({ block: "nearest" });
      }

      function build() {
        menu.innerHTML = "";
        items = [];
        var current = sel.selectedIndex;
        function addOption(o, idx) {
          var row = document.createElement("div");
          row.className = "dd-option" + (o.disabled ? " disabled" : "") + (idx === current ? " selected" : "");
          var txt = document.createElement("span");
          txt.textContent = o.text;
          row.appendChild(txt);
          menu.appendChild(row);
          if (o.disabled) return;
          var entry = { index: idx, el: row };
          items.push(entry);
          row.addEventListener("mousemove", function () { setActive(items.indexOf(entry), false); });
          row.addEventListener("click", function () { choose(idx); });
        }
        var idx = 0;
        Array.prototype.forEach.call(sel.children, function (child) {
          if (child.tagName === "OPTGROUP") {
            var g = document.createElement("div");
            g.className = "dd-group";
            g.textContent = child.label;
            menu.appendChild(g);
            Array.prototype.forEach.call(child.children, function (o) { addOption(o, idx++); });
          } else {
            addOption(child, idx++);
          }
        });
        if (!menu.children.length) {
          var empty = document.createElement("div");
          empty.className = "dd-empty";
          empty.textContent = "No options";
          menu.appendChild(empty);
        }
        var sIdx = items.findIndex(function (it) { return it.index === current; });
        setActive(sIdx >= 0 ? sIdx : 0, false);
      }

      function place() {
        var r = trigger.getBoundingClientRect();
        var room = window.innerHeight - r.bottom - 12;
        var above = r.top - 12;
        var maxH = 300;
        var flip = room < Math.min(maxH, menu.scrollHeight) && above > room;
        menu.style.minWidth = r.width + "px";
        menu.style.maxHeight = Math.max(120, Math.min(maxH, flip ? above : room)) + "px";
        menu.style.left = Math.max(8, Math.min(r.left, window.innerWidth - menu.offsetWidth - 8)) + "px";
        if (flip) {
          menu.style.top = "auto";
          menu.style.bottom = (window.innerHeight - r.top + 6) + "px";
          menu.classList.add("flip");
        } else {
          menu.style.bottom = "auto";
          menu.style.top = (r.bottom + 6) + "px";
          menu.classList.remove("flip");
        }
      }

      var dd = {
        root: root,
        menu: menu,
        open: function () {
          if (sel.disabled) return;
          if (openDD && openDD !== dd) openDD.close(false);
          build();
          document.body.appendChild(menu);
          menu.classList.remove("show");
          place();
          requestAnimationFrame(function () { menu.classList.add("show"); });
          root.classList.add("open");
          sel._ddOpen = true;
          openDD = dd;
          if (items[active]) items[active].el.scrollIntoView({ block: "nearest" });
        },
        close: function (refocus) {
          if (openDD !== dd) return;
          root.classList.remove("open");
          sel._ddOpen = false;
          menu.classList.remove("show");
          if (menu.parentNode) menu.parentNode.removeChild(menu);
          openDD = null;
          if (refocus) sel.focus();
        },
      };
      sel._dd = dd;

      function choose(idx) {
        var changed = sel.selectedIndex !== idx;
        indexDesc.set.call(sel, idx);
        refresh();
        dd.close(true);
        if (changed) sel.dispatchEvent(new Event("change", { bubbles: true }));
      }

      // The menu must never steal focus from the (hidden) native select.
      menu.addEventListener("mousedown", function (e) { e.preventDefault(); });

      trigger.addEventListener("click", function () {
        if (sel.disabled) return;
        sel.focus();
        if (openDD === dd) dd.close(true); else dd.open();
      });

      sel.addEventListener("keydown", function (e) {
        if (sel.disabled) return;
        if (openDD === dd) {
          if (e.key === "ArrowDown") { e.preventDefault(); setActive(active + 1, true); }
          else if (e.key === "ArrowUp") { e.preventDefault(); setActive(active - 1, true); }
          else if (e.key === "Home") { e.preventDefault(); setActive(0, true); }
          else if (e.key === "End") { e.preventDefault(); setActive(items.length - 1, true); }
          else if (e.key === "Enter" || e.key === " ") { e.preventDefault(); if (items[active]) choose(items[active].index); }
          else if (e.key === "Escape") { e.preventDefault(); dd.close(true); }
          else if (e.key === "Tab") { dd.close(false); }
          else if (e.key.length === 1) {
            // type-ahead within the open menu
            var ch = e.key.toLowerCase();
            for (var k = 1; k <= items.length; k++) {
              var cand = (active + k) % items.length;
              if (items[cand].el.textContent.replace(/^[✓⬇]\s*/, "").toLowerCase().indexOf(ch) === 0) { setActive(cand, true); break; }
            }
          }
          return;
        }
        // Closed: native arrow/type-ahead behaviour stays (works with
        // assistive tech); only the "open the popup" keys are redirected.
        if (e.key === "Enter" || e.key === " " || (e.altKey && e.key === "ArrowDown")) {
          e.preventDefault();
          dd.open();
        }
      });
      sel.addEventListener("blur", function () { if (openDD === dd) dd.close(false); });
      sel.addEventListener("change", refresh);

      // Programmatic writes (applyGenericValue, model list sync, ...) go
      // through these setters without firing any event — keep the label in step.
      Object.defineProperty(sel, "value", {
        configurable: true,
        get: function () { return valueDesc.get.call(this); },
        set: function (v) { valueDesc.set.call(this, v); refresh(); },
      });
      Object.defineProperty(sel, "selectedIndex", {
        configurable: true,
        get: function () { return indexDesc.get.call(this); },
        set: function (v) { indexDesc.set.call(this, v); refresh(); },
      });
      new MutationObserver(refresh).observe(sel, {
        childList: true, subtree: true, characterData: true,
        attributes: true, attributeFilter: ["disabled", "label"],
      });
      refresh();
    });
  }

  var COLLAPSED_KEY = "axiom_collapsed_groups";

  // ---------------------------------------------------------------------
  // LAN speed test (Other tab) — browser <-> Axiom host over the same HTTP
  // server/token as everything else. Ping x12 (first is warm-up), then a
  // download and an upload ramp (2 -> 8 -> 16 MiB, stopping the ramp once a
  // pass takes over 2 s). Bounded: <= ~26 MiB each way. Server side:
  // core/speedtest.py + /api/speedtest/*.
  // ---------------------------------------------------------------------
  var SPEED_RAMP_MB = [2, 8, 16];
  var SPEED_PASS_BUDGET_MS = 2000;

  function speedMbps(bytes, ms) {
    return ms > 0 ? (bytes * 8) / (ms / 1000) / 1e6 : 0;
  }

  function speedFmt(mbit) {
    return mbit.toFixed(mbit >= 100 ? 0 : 1) + " Mbit/s (" + (mbit / 8).toFixed(mbit >= 800 ? 0 : 1) + " MB/s)";
  }

  function speedSummarize(samples) {
    var clean = samples.filter(function (v) { return isFinite(v) && v >= 0; });
    if (!clean.length) return { count: 0, min: 0, median: 0, max: 0, jitter: 0 };
    var sorted = clean.slice().sort(function (a, b) { return a - b; });
    var mid = sorted.length >> 1;
    var median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    var diffs = [];
    for (var i = 1; i < clean.length; i++) diffs.push(Math.abs(clean[i] - clean[i - 1]));
    var jitter = diffs.length ? diffs.reduce(function (a, b) { return a + b; }, 0) / diffs.length : 0;
    return { count: clean.length, min: sorted[0], median: median, max: sorted[sorted.length - 1], jitter: jitter };
  }

  function speedFetch(path, opts) {
    opts = opts || {};
    var headers = authHeaders();
    if (opts.rawBody) delete headers["Content-Type"];     // octet-stream upload, not JSON
    opts.headers = headers;
    opts.cache = "no-store";
    delete opts.rawBody;
    return fetch(path, opts).then(function (res) {
      if (res.status === 401) throw new Error("Unauthorized — check the access token");
      if (!res.ok) throw new Error("Server error (HTTP " + res.status + ")");
      return res;
    }, function (err) {
      if (err && err.name === "AbortError") throw err;
      throw new Error("Cannot reach the Axiom host");
    });
  }

  // UDP "ping": the browser cannot send UDP, so this reports the quality of the real UDP video
  // stream Axiom is receiving (relative timings measured on the host; no one-way delay).
  function speedUdpStats(signal) {
    return speedFetch("/api/status", { signal: signal }).then(function (r) { return r.json(); }).then(function (s) {
      if (s.screenshot_method !== "udp") return null;
      return s.udp_stats || { active: false };
    }).catch(function (err) { if (err && err.name === "AbortError") throw err; return null; });
  }

  function speedUdpLine(u) {
    if (u === null) return "UDP stream: set Capture Method to UDP to measure the video stream";
    if (!u.active) return "UDP stream: no frames arriving";
    return "UDP stream: frames every " + u.interval_ms.toFixed(1) + " ms (jitter " + u.jitter_ms.toFixed(1) +
      " ms, min " + u.interval_min_ms.toFixed(1) + ", max " + u.interval_max_ms.toFixed(1) + ")\n" +
      "  packets of one frame arrive within " + u.assembly_ms.toFixed(1) + " ms (worst " + u.assembly_max_ms.toFixed(1) +
      " ms); lost " + u.lost_frames + " of " + (u.lost_frames + u.frames) + " frames (" + u.loss_pct.toFixed(1) +
      " %), reordered " + u.reordered;
  }

  function speedPing(count, signal) {
    var samples = [];
    samples.proc = [];                  // server-side handling time per sample (ms), for "host vs network"
    function one(i) {
      if (i >= count) return Promise.resolve(samples);
      var t0 = performance.now();
      return speedFetch("/api/speedtest/ping", { signal: signal }).then(function (r) { return r.json(); }).then(function (body) {
        if (i > 0) {                    // sample 0 = connection warm-up
          samples.push(performance.now() - t0);
          if (body && isFinite(body.proc_ms)) samples.proc.push(body.proc_ms);
        }
        return one(i + 1);
      });
    }
    return one(0);
  }

  // One download pass -> {bytes, ms}. Timing starts at the first received chunk so
  // request latency doesn't count against throughput. If the whole body shows up
  // as (nearly) one chunk -- a fast link or a buffering browser -- there is nothing
  // to time after the first chunk, so fall back to the full request time instead
  // of reporting 0.
  function speedDownloadPass(mb, signal) {
    var tReq = performance.now();
    return speedFetch("/api/speedtest/download?mb=" + mb, { signal: signal }).then(function (res) {
      if (!res.body || !res.body.getReader) {           // very old browser: whole-body fallback
        return res.arrayBuffer().then(function (buf) { return { bytes: buf.byteLength, ms: performance.now() - tReq }; });
      }
      var reader = res.body.getReader();
      var total = 0, first = 0, tStart = null;
      function pump() {
        return reader.read().then(function (r) {
          if (r.done) {
            var now = performance.now();
            var rest = total - first;
            if (tStart === null || rest < total * 0.1 || now - tStart < 2) return { bytes: total, ms: now - tReq };
            return { bytes: rest, ms: now - tStart };
          }
          if (tStart === null) { tStart = performance.now(); first = r.value.length; }
          total += r.value.length;
          return pump();
        });
      }
      return pump();
    });
  }

  function speedUploadPass(mb, signal) {
    var size = mb * 1024 * 1024;
    var block = new Uint8Array(65536);
    if (window.crypto && crypto.getRandomValues) crypto.getRandomValues(block);
    var data = new Uint8Array(size);
    for (var o = 0; o < size; o += block.length) data.set(block.subarray(0, Math.min(block.length, size - o)), o);
    var t0 = performance.now();
    return speedFetch("/api/speedtest/upload", { method: "POST", body: new Blob([data]), signal: signal, rawBody: true })
      .then(function (r) { return r.json(); })
      .then(function (j) { return { bytes: j.bytes || size, ms: performance.now() - t0 }; });
  }

  // Runs the ramp and resolves the best (fastest) pass in Mbit/s.
  function speedRamp(passFn, signal, onPass) {
    var best = 0;
    function next(i) {
      if (i >= SPEED_RAMP_MB.length) return Promise.resolve(best);
      return passFn(SPEED_RAMP_MB[i], signal).then(function (res) {
        var m = speedMbps(res.bytes, res.ms);
        if (m > best) best = m;
        if (onPass) onPass(m, SPEED_RAMP_MB[i]);
        return res.ms > SPEED_PASS_BUDGET_MS ? best : next(i + 1);   // slow link: stop ramping
      });
    }
    return next(0);
  }

  function initSpeedTest() {
    var btn = document.getElementById("other-speed-btn");
    var stopBtn = document.getElementById("other-speed-stop");
    var stageEl = document.getElementById("other-speed-stage");
    var resultEl = document.getElementById("other-speed-result");
    if (!btn || !stageEl || !resultEl) return;
    var idle = stageEl.textContent;
    var controller = null;

    function setRunning(on) {
      btn.hidden = on;
      if (stopBtn) stopBtn.hidden = !on;
    }

    btn.addEventListener("click", function () {
      if (controller) return;
      controller = (typeof AbortController !== "undefined") ? new AbortController() : { signal: undefined, abort: function () {} };
      var signal = controller.signal;
      var out = { ping: null, proc: null, down: null, up: null, udp: undefined };
      resultEl.hidden = true;
      setRunning(true);

      function render() {
        var lines = [];
        if (out.ping) lines.push("Ping: " + out.ping.median.toFixed(1) + " ms median (min " + out.ping.min.toFixed(1) +
          ", max " + out.ping.max.toFixed(1) + ", jitter " + out.ping.jitter.toFixed(1) + " ms)" +
          (out.proc ? "\n  Axiom handled each request in " + out.proc.median.toFixed(2) + " ms; the rest is browser, network and host scheduling" : ""));
        if (out.down !== null) lines.push("Download: " + speedFmt(out.down));
        if (out.up !== null) lines.push("Upload: " + speedFmt(out.up));
        if (out.udp !== undefined) lines.push(speedUdpLine(out.udp));
        resultEl.textContent = lines.join("\n");
        resultEl.hidden = !lines.length;
      }

      stageEl.textContent = "Testing ping…";
      speedPing(12, signal).then(function (samples) {
        out.ping = speedSummarize(samples);
        out.proc = samples.proc.length ? speedSummarize(samples.proc) : null;
        render();
        stageEl.textContent = "Testing download…";
        return speedRamp(speedDownloadPass, signal, function (m, mb) { out.down = m; render(); });
      }).then(function (down) {
        out.down = down; render();
        stageEl.textContent = "Testing upload…";
        return speedRamp(speedUploadPass, signal, function (m) { out.up = m; render(); });
      }).then(function (up) {
        out.up = up; render();
        return speedUdpStats(signal);
      }).then(function (udp) {
        out.udp = udp; render();
        stageEl.textContent = "Done — browser to Axiom host over one HTTP stream. Wi-Fi/router load or this browser can cap the result.";
      }).catch(function (err) {
        stageEl.textContent = (err && err.name === "AbortError") ? "Stopped." : "Failed: " + (err && err.message || err);
        render();
      }).then(function () {
        controller = null;
        setRunning(false);
      });
    });

    if (stopBtn) stopBtn.addEventListener("click", function () { if (controller) controller.abort(); });
    stageEl.setAttribute("data-idle", idle);
  }

  // Set once the search box exists; lets the advanced switch re-run the filter.
  var searchRefresh = null;
  var ADVANCED_CACHE_KEY = "axiom_show_advanced";

  // "Show advanced settings": cards carrying data-advanced are hidden by CSS (body:not(.show-advanced))
  // while it is off. Display only — hidden fields keep their values and are still read/synced, exactly
  // like conditionally hidden ones. The flag itself lives on the host (config show_advanced_settings),
  // so the Qt app and this page always agree.
  function initAdvancedSettings() {
    var toggle = document.getElementById("adv-toggle");
    // The attribute is written on the control; the card is what hides.
    Array.prototype.forEach.call(document.querySelectorAll(".main [data-advanced]"), function (el) {
      if (el.classList.contains("field-card")) return;
      var card = el.closest(".field-card");
      if (card) card.setAttribute("data-advanced", "1");
    });
    var modules = Array.prototype.slice.call(document.querySelectorAll(".main .module"));
    var gated = modules.filter(function (m) { return m.querySelector(".field-card[data-advanced]"); });

    function refreshEmptyModules() {
      gated.forEach(function (m) {
        // A section whose cards are all advanced would show only its title.
        var basic = m.querySelector(".field-card:not([data-advanced])");
        m.classList.toggle("adv-empty", !advancedOn && !basic);
      });
    }
    function setAdvanced(on) {
      advancedOn = !!on;
      document.body.classList.toggle("show-advanced", advancedOn);
      if (toggle) toggle.checked = advancedOn;
      refreshEmptyModules();
      if (searchRefresh) searchRefresh();
      var cached = null;
      try { cached = window.localStorage; } catch (e) { cached = null; }
      if (cached) { try { cached.setItem(ADVANCED_CACHE_KEY, advancedOn ? "1" : "0"); } catch (e2) { /* private mode */ } }
    }
    // Last known value first (no flash on reload); the first status poll reconfirms it.
    var last = null;
    try { last = window.localStorage.getItem(ADVANCED_CACHE_KEY); } catch (e) { last = null; }
    setAdvanced(last === null ? true : last === "1");
    setAdvancedFromServer = function (on) {
      if (document.activeElement === toggle) return;       // never fight the user's own click
      if (on !== advancedOn) setAdvanced(on);
    };
    if (toggle) {
      toggle.addEventListener("change", function () {
        var want = toggle.checked;
        setAdvanced(want);
        advancedPending = true;
        api("POST", "/api/settings/visuals", { show_advanced_settings: want })
          .then(function (data) {
            if (!data || data.ok === false) throw new Error(data && data.reason ? String(data.reason).replace(/_/g, " ") : "rejected");
          })
          .catch(function (err) {
            setAdvanced(!want);
            notify("Could not save the advanced settings switch: " + (err && err.message ? err.message : "request failed"), true);
          })
          .then(function () { advancedPending = false; });
      });
    }
  }

  function initUiEnhancements() {
    var i;

    // Tab panels: roles + labelled-by (tab buttons already carry theirs).
    panels.forEach(function (panel) {
      panel.setAttribute("role", "tabpanel");
      panel.setAttribute("aria-labelledby", "tab-" + panel.id.replace("panel-", ""));
    });

    // Arrow-key / Home / End navigation across tabs.
    var tabList = Array.prototype.slice.call(tabButtons);
    tabList.forEach(function (btn, idx) {
      btn.addEventListener("keydown", function (e) {
        var target = -1;
        if (e.key === "ArrowRight") target = (idx + 1) % tabList.length;
        else if (e.key === "ArrowLeft") target = (idx - 1 + tabList.length) % tabList.length;
        else if (e.key === "Home") target = 0;
        else if (e.key === "End") target = tabList.length - 1;
        if (target < 0) return;
        e.preventDefault();
        activateTab(tabList[target].getAttribute("data-tab"));
        tabList[target].focus();
      });
    });

    // Accessible names: every field control is named by its card's title
    // (descriptions go in aria-describedby) instead of being an unlabeled
    // checkbox/number/select to a screen reader.
    var cards = document.querySelectorAll(".field-card");
    var uid = 0;
    Array.prototype.forEach.call(cards, function (card) {
      var title = card.querySelector(".field-title");
      if (!title) return;
      if (!title.id) title.id = "ft-" + (++uid);
      var desc = card.querySelector(".field-desc");
      if (desc && !desc.id) desc.id = "fd-" + (++uid);
      Array.prototype.forEach.call(card.querySelectorAll("input, select, textarea"), function (ctl) {
        if (ctl.hasAttribute("aria-label") || ctl.hasAttribute("aria-labelledby")) return;
        if (ctl.type === "file" || ctl.type === "hidden") return;
        ctl.setAttribute("aria-labelledby", title.id);
        if (desc) ctl.setAttribute("aria-describedby", desc.id);
      });
    });

    // Collapsible groups (title click / Enter / Space), state remembered.
    var collapsed = getLocalJSON(COLLAPSED_KEY) || {};
    var groupTitles = document.querySelectorAll(".main .field-group-title");
    Array.prototype.forEach.call(groupTitles, function (titleEl) {
      var group = titleEl.nextElementSibling;
      if (!group || !group.classList.contains("field-group")) return;
      var panel = titleEl.closest(".panel");
      var gkey = (panel ? panel.id : "") + "::" + titleEl.textContent.trim();
      titleEl.classList.add("collapsible");
      titleEl.setAttribute("role", "button");
      titleEl.tabIndex = 0;
      function setCollapsed(on) {
        titleEl.classList.toggle("collapsed", on);
        group.classList.toggle("collapsed", on);
        titleEl.setAttribute("aria-expanded", on ? "false" : "true");
      }
      // Saved choice wins; otherwise "advanced" groups marked
      // data-default-collapsed start folded away.
      var byDefault = titleEl.getAttribute("data-default-collapsed") === "1";
      setCollapsed(Object.prototype.hasOwnProperty.call(collapsed, gkey) ? !!collapsed[gkey] : byDefault);
      function toggle() {
        var on = !titleEl.classList.contains("collapsed");
        setCollapsed(on);
        collapsed[gkey] = on;  // explicit true/false so an opened default-collapsed group stays open
        setLocalJSON(COLLAPSED_KEY, collapsed);
      }
      titleEl.addEventListener("click", toggle);
      titleEl.addEventListener("keydown", function (e) {
        if (e.key === "Enter" || e.key === " ") { e.preventDefault(); toggle(); }
      });
    });

    // Settings search across every panel.
    var searchInput = document.getElementById("settings-search");
    var searchCount = document.getElementById("search-count");
    function cardText(card) {
      var t = card.querySelector(".field-title");
      var d = card.querySelector(".field-desc");
      if (!t && !d) return card.textContent.toLowerCase();
      return ((t ? t.textContent : "") + " " + (d ? d.textContent : "")).toLowerCase();
    }
    function deckText(deck) {
      var parts = [deck.getAttribute("data-label") || ""];
      Array.prototype.forEach.call(deck.querySelectorAll("button, [title], input[placeholder], .field-title"), function (el) {
        parts.push(el.textContent || "", el.getAttribute("title") || "", el.getAttribute("placeholder") || "");
      });
      return parts.join(" ").toLowerCase();
    }
    function applySearch() {
      var q = searchInput.value.trim().toLowerCase();
      document.body.classList.toggle("searching", !!q);
      var matches = 0;
      var advancedMatches = 0;     // hits that are only hidden by the Show advanced settings switch
      Array.prototype.forEach.call(document.querySelectorAll(".main .field-card"), function (card) {
        var hit = !q || cardText(card).indexOf(q) !== -1;
        card.classList.toggle("search-hidden", !hit);
        if (q && hit && !card.classList.contains("hidden") && !card.closest(".hidden")) {
          if (advancedOn || !card.hasAttribute("data-advanced")) matches++;
          else advancedMatches++;
        }
      });
      function hasVisibleCard(el) {
        var list = el.querySelectorAll(".field-card:not(.search-hidden):not(.hidden)");
        for (var k = 0; k < list.length; k++) {
          if (advancedOn || !list[k].hasAttribute("data-advanced")) return true;
        }
        return false;
      }
      // Panels are: title, an optional .deck (primary controls) and a
      // .modules wrapper of <section class="module"> blocks, each a group
      // title + group (see index.html). While filtering, a module or deck
      // with no matching card is hidden, as are hints/logs between cards.
      Array.prototype.forEach.call(document.querySelectorAll(".main .panel"), function (panel) {
        var anyInPanel = false;
        Array.prototype.forEach.call(panel.querySelectorAll(".deck, .module"), function (box) {
          var vis = hasVisibleCard(box);
          // A deck holds the tab's primary controls (model picker, Start/Stop AI, capture method), which are
          // not field cards: match them by their label, button captions and tooltips instead of hiding them.
          if (!vis && q && box.classList.contains("deck") && deckText(box).indexOf(q) !== -1) vis = true;
          box.classList.toggle("search-hidden", !!q && !vis);
          if (vis && !box.closest(".hidden")) anyInPanel = true;
        });
        Array.prototype.forEach.call(panel.querySelectorAll(".field-group"), function (g) {
          g.classList.toggle("search-hidden", !!q && !hasVisibleCard(g));
        });
        Array.prototype.forEach.call(panel.querySelectorAll(".hint, textarea, .progress-bar, .preview-img, .module-note"), function (el) {
          if (el.closest(".field-card")) return;
          el.classList.toggle("search-hidden", !!q);
        });
        panel.classList.toggle("search-hidden", !!q && !anyInPanel);
      });
      var countText = q ? (matches + (matches === 1 ? " setting" : " settings") + " match") : "";
      if (q && advancedMatches) {
        countText += " · " + advancedMatches + " more in advanced settings (turn on Show advanced settings)";
      }
      searchCount.textContent = countText;
    }
    searchRefresh = applySearch;
    if (searchInput) {
      searchInput.addEventListener("input", applySearch);
      searchInput.addEventListener("keydown", function (e) {
        if (e.key === "Escape") { searchInput.value = ""; applySearch(); }
      });
    }

    initAdvancedSettings();

    // Phones: the sidebar (token/status/presets) is collapsible so settings
    // aren't buried underneath it. Starts collapsed once a token exists.
    var sidebar = document.getElementById("sidebar");
    var sidebarToggle = document.getElementById("sidebar-toggle");
    if (sidebar && sidebarToggle) {
      var narrow = window.matchMedia ? window.matchMedia("(max-width: 760px)") : null;
      function setSidebar(open) {
        sidebar.classList.toggle("collapsed", !open);
        sidebarToggle.setAttribute("aria-expanded", open ? "true" : "false");
      }
      setSidebar(!(narrow && narrow.matches && getToken()));
      sidebarToggle.addEventListener("click", function () {
        setSidebar(sidebar.classList.contains("collapsed"));
      });
    }

    enhanceSelects();

    // Interface style (Other tab): two looks over the same page, chosen per
    // browser. The <head> script applies the saved one before first paint.
    var UI_STYLE_KEY = "axiom_ui_style";
    function applyUiStyle(style, announce) {
      if (style !== "instrument" && style !== "studio") style = "studio";
      document.documentElement.setAttribute("data-ui", style);
      try { localStorage.setItem(UI_STYLE_KEY, style); } catch (e) { /* storage unavailable */ }
      Array.prototype.forEach.call(document.querySelectorAll('input[name="ui-style"]'), function (r) {
        r.checked = r.value === style;
      });
      if (announce) notify("Interface style: " + (style === "studio" ? "Studio" : "Instrument"));
    }
    applyUiStyle(document.documentElement.getAttribute("data-ui"), false);
    Array.prototype.forEach.call(document.querySelectorAll('input[name="ui-style"]'), function (r) {
      r.addEventListener("change", function () { if (r.checked) applyUiStyle(r.value, true); });
    });

    // Other tab — client-only tools (nothing here changes server state).
    var testBtn = document.getElementById("other-test-btn");
    var latencyEl = document.getElementById("other-latency");
    if (testBtn) {
      testBtn.addEventListener("click", function () {
        latencyEl.textContent = "Testing…";
        // The light ping route (not /api/status, which builds the whole status dict), so the
        // number is the connection, not Axiom's own work. First sample is the connection warm-up.
        speedPing(6, undefined).then(function (samples) {
          var m = speedSummarize(samples);
          latencyEl.textContent = "Connected — round trip " + m.median.toFixed(1) + " ms median (min " +
            m.min.toFixed(1) + ", max " + m.max.toFixed(1) + ")";
        }).catch(function (err) {
          latencyEl.textContent = "Failed: " + err.message;
        });
      });
    }
    initSpeedTest();
    var clearBtn = document.getElementById("other-clear-token-btn");
    if (clearBtn) {
      clearBtn.addEventListener("click", function () {
        setToken("");
        tokenInput.value = "";
        authState = "unknown";
        notify("Saved token cleared from this browser");
        poll();
      });
    }
    var resetUiBtn = document.getElementById("other-reset-ui-btn");
    if (resetUiBtn) {
      resetUiBtn.addEventListener("click", function () {
        try {
          localStorage.removeItem(COLLAPSED_KEY);
          localStorage.removeItem(LAST_TAB_KEY);
        } catch (e) { /* storage unavailable — nothing to reset */ }
        collapsed = {};
        Array.prototype.forEach.call(groupTitles, function (titleEl) {
          var g = titleEl.nextElementSibling;
          if (!g || !g.classList.contains("field-group")) return;
          var on = titleEl.getAttribute("data-default-collapsed") === "1";
          titleEl.classList.toggle("collapsed", on);
          g.classList.toggle("collapsed", on);
          titleEl.setAttribute("aria-expanded", on ? "false" : "true");
        });
        applyUiStyle("studio", false);
        notify("Page layout reset");
      });
    }
  }

  initUiEnhancements();
  poll();
})();
