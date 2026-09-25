/* RockMap dashboard client-side behaviour (no build step, no dependencies). */
(function () {
  "use strict";

  // ---------------------------------------------------------------- layer tabs + opacity
  document.querySelectorAll("[data-viewer]").forEach(function (root) {
    var base = root.querySelector("img.base");
    var overlay = root.querySelector("img.overlay");
    var tabs = document.querySelectorAll('[data-base-for="' + root.id + '"] button');
    tabs.forEach(function (btn) {
      btn.addEventListener("click", function () {
        tabs.forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        base.src = btn.dataset.src;
      });
    });
    var slider = document.querySelector('[data-opacity-for="' + root.id + '"]');
    if (slider && overlay) {
      var apply = function () { overlay.style.opacity = slider.value / 100; };
      slider.addEventListener("input", apply);
      apply();
    }
  });

  // ---------------------------------------------------------------- region (window) selection
  var sel = document.getElementById("region-viewer");
  if (sel) {
    var canvas = sel.querySelector("canvas");
    var fullW = +sel.dataset.width, fullH = +sel.dataset.height;
    var input = document.getElementById("window-input");
    var label = document.getElementById("window-label");
    var start = null, rect = null;
    var ctx = canvas.getContext("2d");

    function resize() {
      canvas.width = canvas.clientWidth; canvas.height = canvas.clientHeight; draw();
    }
    function pos(e) {
      var r = canvas.getBoundingClientRect();
      var p = e.touches ? e.touches[0] : e;
      return { x: Math.max(0, Math.min(r.width, p.clientX - r.left)), y: Math.max(0, Math.min(r.height, p.clientY - r.top)) };
    }
    function draw() {
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      if (!rect) return;
      ctx.fillStyle = "rgba(0,0,0,.45)";
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      ctx.clearRect(rect.x, rect.y, rect.w, rect.h);
      ctx.strokeStyle = "#ffcc00"; ctx.lineWidth = 2; ctx.setLineDash([6, 4]);
      ctx.strokeRect(rect.x, rect.y, rect.w, rect.h);
    }
    function commit() {
      if (!rect || rect.w < 4 || rect.h < 4) { clear(); return; }
      var sx = fullW / canvas.width, sy = fullH / canvas.height;
      var win = [Math.round(rect.x * sx), Math.round(rect.y * sy), Math.round(rect.w * sx), Math.round(rect.h * sy)];
      input.value = win.join(",");
      label.textContent = "Region: columns " + win[0] + "–" + (win[0] + win[2]) + ", rows " + win[1] + "–" + (win[1] + win[3]) +
        " (" + win[2] + " × " + win[3] + " px)";
    }
    function clear() { rect = null; input.value = ""; label.textContent = "Whole scene"; draw(); }
    function down(e) { e.preventDefault(); start = pos(e); rect = { x: start.x, y: start.y, w: 0, h: 0 }; }
    function move(e) {
      if (!start) return; e.preventDefault();
      var p = pos(e);
      rect = { x: Math.min(p.x, start.x), y: Math.min(p.y, start.y), w: Math.abs(p.x - start.x), h: Math.abs(p.y - start.y) };
      draw();
    }
    function up() { if (start) { start = null; commit(); } }
    canvas.addEventListener("mousedown", down); canvas.addEventListener("touchstart", down, { passive: false });
    window.addEventListener("mousemove", move); canvas.addEventListener("touchmove", move, { passive: false });
    window.addEventListener("mouseup", up); window.addEventListener("touchend", up);
    var clr = document.getElementById("window-clear");
    if (clr) clr.addEventListener("click", clear);
    window.addEventListener("resize", resize);
    var img = sel.querySelector("img.base");
    if (img.complete) resize(); else img.addEventListener("load", resize);
  }

  // ---------------------------------------------------------------- job polling
  var job = document.getElementById("job-progress");
  if (job) {
    var bar = job.querySelector(".progress > span");
    var msg = job.querySelector(".msg");
    var poll = function () {
      fetch(job.dataset.api).then(function (r) { return r.json(); }).then(function (j) {
        bar.style.width = Math.round(j.progress * 100) + "%";
        msg.textContent = j.message || j.status;
        var logEl = job.querySelector("[data-log]");
        if (logEl && j.log) { logEl.textContent = j.log; logEl.scrollTop = logEl.scrollHeight; }
        if (["done", "failed", "cancelled"].indexOf(j.status) >= 0) { window.location.reload(); }
        else { setTimeout(poll, 1000); }
      }).catch(function () { setTimeout(poll, 3000); });
    };
    poll();
  }

  // ---------------------------------------------------------------- web map (Leaflet, optional)
  var mapEl = document.getElementById("leaflet-map");
  if (mapEl && window.L) {
    var b = JSON.parse(mapEl.dataset.bounds); // [west, south, east, north]
    var bounds = [[b[1], b[0]], [b[3], b[2]]];
    var map = L.map(mapEl).fitBounds(bounds);
    var osm = L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
      maxZoom: 18, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
    var sat = L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}", {
      maxZoom: 18, attribution: "Imagery &copy; Esri" });
    var overlays = {};
    JSON.parse(mapEl.dataset.layers).forEach(function (l, i) {
      var layer = L.imageOverlay(l.src, bounds, { opacity: l.opacity || 0.75 });
      if (i === 0) layer.addTo(map);
      overlays[l.name] = layer;
    });
    L.control.layers({ "OpenStreetMap": osm, "Satellite (Esri)": sat }, overlays, { collapsed: false }).addTo(map);
    L.rectangle(bounds, { color: "#ffcc00", weight: 2, fill: false }).addTo(map);
    var op = document.querySelector('[data-opacity-for="leaflet-map"]');
    if (op) op.addEventListener("input", function () {
      Object.keys(overlays).forEach(function (k) { overlays[k].setOpacity(op.value / 100); });
    });
    // Leaflet needs a size refresh when shown from a hidden tab
    document.querySelectorAll("[data-show]").forEach(function (btn) {
      btn.addEventListener("click", function () { setTimeout(function () { map.invalidateSize(); }, 50); });
    });
  } else if (mapEl) {
    mapEl.innerHTML = '<p class="empty">Web map unavailable (Leaflet could not be loaded — are you offline?). Use the image viewer instead.</p>';
  }

  // ---------------------------------------------------------------- simple show/hide tabs
  document.querySelectorAll("[data-tabgroup]").forEach(function (group) {
    var buttons = group.querySelectorAll("[data-show]");
    buttons.forEach(function (btn) {
      btn.addEventListener("click", function () {
        buttons.forEach(function (b) {
          b.classList.toggle("active", b === btn);
          var el = document.getElementById(b.dataset.show);
          if (el) el.hidden = b !== btn;
        });
      });
    });
  });

  // ---------------------------------------------------------------- CNN training curve (inline SVG)
  document.querySelectorAll("svg[data-history]").forEach(function (svg) {
    var h = JSON.parse(svg.dataset.history);
    if (!h.length) return;
    var W = 560, H = 220, L = 40, R = 12, T = 12, B = 28;
    svg.setAttribute("viewBox", "0 0 " + W + " " + H);
    var ns = "http://www.w3.org/2000/svg";
    function el(tag, attrs, text) {
      var e = document.createElementNS(ns, tag);
      for (var k in attrs) e.setAttribute(k, attrs[k]);
      if (text !== undefined) e.textContent = text;
      svg.appendChild(e); return e;
    }
    var n = h.length;
    var lo = Math.min.apply(null, h.map(function (r) { return Math.min(r.train_acc, r.val_acc || 1); }));
    lo = Math.max(0, Math.floor(lo * 10) / 10);
    var x = function (i) { return L + (n === 1 ? 0 : i / (n - 1)) * (W - L - R); };
    var y = function (v) { return T + (1 - (v - lo) / (1 - lo)) * (H - T - B); };
    for (var g = lo; g <= 1.0001; g += (1 - lo) / 4) {
      el("line", { x1: L, x2: W - R, y1: y(g), y2: y(g), class: "grid" });
      el("text", { x: L - 6, y: y(g) + 4, "text-anchor": "end" }, (g * 100).toFixed(0) + "%");
    }
    el("text", { x: (L + W - R) / 2, y: H - 6, "text-anchor": "middle" }, "epoch (1–" + n + ")");
    var series = [["train_acc", "#8a8f98", "train"], ["val_acc", "#b5562b", "validation"]];
    series.forEach(function (s, si) {
      if (h[0][s[0]] === undefined) return;
      var d = h.map(function (r, i) { return (i ? "L" : "M") + x(i).toFixed(1) + "," + y(r[s[0]]).toFixed(1); }).join("");
      el("path", { d: d, fill: "none", stroke: s[1], "stroke-width": 2 });
      el("text", { x: W - R - 4, y: T + 14 + si * 14, "text-anchor": "end", style: "fill:" + s[1] }, s[2]);
    });
  });
})();
