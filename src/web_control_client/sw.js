/* Axiom Web Control service worker.
 *
 * Caches only the *app shell* (page, script, styles, icons) so the page opens instantly and shows a
 * friendly "unreachable" screen when the Axiom host is off. Control data is never cached: every
 * /api/* request (settings, status, the live preview stream, the access-token header) goes straight
 * to the network and is not touched here.
 *
 * Keep SHELL_VERSION in step with the ?v=N asset version in index.html (bump both together).
 * Browsers only run service workers in a secure context (https:// or http://localhost).
 */
"use strict";

var SHELL_VERSION = "27";
var CACHE = "axiom-shell-v" + SHELL_VERSION;
var SHELL = [
  "./",
  "index.html",
  "styles.css?v=" + SHELL_VERSION,
  "app.js?v=" + SHELL_VERSION,
  "manifest.webmanifest",
  "offline.html",
  "icons/icon-192.png",
  "icons/icon-512.png",
  "icons/favicon-32.png"
];

self.addEventListener("install", function (event) {
  // Do not skipWaiting here: the page offers "Update available - reload" and asks for it.
  event.waitUntil(
    caches.open(CACHE).then(function (cache) {
      return Promise.all(SHELL.map(function (url) {
        return cache.add(new Request(url, { cache: "reload" })).catch(function () { /* one miss must not block install */ });
      }));
    })
  );
});

self.addEventListener("activate", function (event) {
  event.waitUntil(
    caches.keys().then(function (keys) {
      return Promise.all(keys.filter(function (k) { return k.indexOf("axiom-shell-") === 0 && k !== CACHE; })
        .map(function (k) { return caches.delete(k); }));
    }).then(function () { return self.clients.claim(); })
  );
});

self.addEventListener("message", function (event) {
  if (event.data && event.data.type === "SKIP_WAITING") self.skipWaiting();
});

function isApi(url) {
  return url.pathname.indexOf("/api/") === 0 || url.pathname === "/api";
}

self.addEventListener("fetch", function (event) {
  var req = event.request;
  if (req.method !== "GET") return;
  var url = new URL(req.url);
  if (url.origin !== self.location.origin || isApi(url)) return;     // network only, never cached

  // Page loads: network first (always the newest shell), cached page / offline screen when unreachable.
  if (req.mode === "navigate") {
    event.respondWith(
      fetch(req).then(function (res) {
        var copy = res.clone();
        caches.open(CACHE).then(function (c) { c.put("index.html", copy); });
        return res;
      }).catch(function () {
        return caches.match("index.html").then(function (hit) { return hit || caches.match("offline.html"); });
      })
    );
    return;
  }

  // Shell assets: stale-while-revalidate (instant, refreshed in the background).
  event.respondWith(
    caches.open(CACHE).then(function (cache) {
      return cache.match(req).then(function (hit) {
        var fresh = fetch(req).then(function (res) {
          if (res && res.ok) cache.put(req, res.clone());
          return res;
        }).catch(function () { return hit; });
        return hit || fresh;
      });
    })
  );
});
