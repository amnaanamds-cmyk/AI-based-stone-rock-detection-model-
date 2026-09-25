/* Region map: tile layers, tile grid, point query, training-area drawing, live job progress. */
(function () {
  "use strict";
  var el = document.getElementById("region-map");
  if (!el || !window.L) return;
  var csrf = document.querySelector('meta[name="csrf-token"]').content;
  var esc = function (v) {
    return String(v === undefined || v === null ? "" : v).replace(/[&<>"']/g, function (c) {
      return { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c];
    });
  };
  var canEdit = el.dataset.canEdit === "1";
  var classes = JSON.parse(el.dataset.classes);
  var colorOf = {}; classes.forEach(function (c) { colorOf[c.id] = c.color; });
  var tileUrl = function (layer) { return el.dataset.tileUrl.replace("LAYER", layer); };
  var available = JSON.parse(el.dataset.layers);

  var map = L.map(el, { preferCanvas: true, zoomControl: true });
  window.rockmapMap = map;
  var bases = {
    "OpenStreetMap": L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
      { maxZoom: 18, attribution: "&copy; OpenStreetMap contributors" }),
    "Esri World Imagery": L.tileLayer("https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
      { maxZoom: 18, attribution: "Imagery &copy; Esri" }),
    "No basemap (offline)": L.layerGroup()
  };
  var labels = { rgb: "Sentinel-2 composite (true colour)", falsecolor: "Sentinel-2 SWIR false colour", hillshade: "DEM hillshade" };
  ["rgb", "falsecolor", "hillshade"].forEach(function (l) {
    if (available.indexOf(l) >= 0) bases[labels[l]] = L.tileLayer(tileUrl(l), { maxZoom: 17, maxNativeZoom: 16, attribution: "Contains modified Copernicus Sentinel data" });
  });
  var baseDefault = available.indexOf("rgb") >= 0 ? bases[labels.rgb] : bases["OpenStreetMap"];
  baseDefault.addTo(map);

  var overlays = {};
  var lith = null;
  if (available.indexOf("lithology") >= 0) {
    lith = L.tileLayer(tileUrl("lithology"), { maxZoom: 17, maxNativeZoom: 16, opacity: 0.8 }).addTo(map);
    overlays["Lithology map"] = lith;
  }
  if (available.indexOf("surface") >= 0) {
    overlays["Surface cover (snow, water, vegetation, shadow)"] = L.tileLayer(tileUrl("surface"), { maxZoom: 17, maxNativeZoom: 16, opacity: 0.85 });
    if (!lith) overlays["Surface cover (snow, water, vegetation, shadow)"].addTo(map);
  }
  if (available.indexOf("confidence") >= 0) overlays["Confidence"] = L.tileLayer(tileUrl("confidence"), { maxZoom: 17, maxNativeZoom: 16, opacity: 0.7 });

  var statusColor = function (p) { return p.error ? "#c62828" : p.classified ? "#2e7d32" : p.acquired ? "#e0a458" : "#9aa0a6"; };
  var grid = L.geoJSON(null, {
    style: function (f) { return { color: statusColor(f.properties), weight: 1, fillOpacity: lith ? 0 : 0.12 }; },
    onEachFeature: function (f, layer) {
      var p = f.properties;
      layer.bindTooltip(esc("Tile " + p.key + (p.acquired ? " · " + (p.scenes || 0) + " scenes, " + Math.round((p.clear || 0) * 100) + "% clear" : " · pending") +
        (p.classified ? " · classified" : "") + (p.error ? " · " + p.error : "")), { sticky: true });
    }
  }).addTo(map);
  overlays["Processing tiles"] = grid;
  var aoi = L.geoJSON(null, { style: { color: "#b5562b", weight: 2, fill: false, dashArray: "6 4" } }).addTo(map);
  overlays["Area of interest"] = aoi;
  var ann = L.geoJSON(null, {
    style: function (f) { return { color: "#111", weight: 1.5, fillColor: f.properties.color, fillOpacity: 0.55 }; },
    onEachFeature: function (f, layer) {
      var p = f.properties;
      var html = "<b>Training area</b><br><span class='swatch' style='background:" + esc(p.color) + "'></span>" + esc(p["class"]) +
        "<br><span class='muted'>" + esc(p.author) + " · " + esc(p.created) + "</span>";
      if (canEdit) html += "<br><button class='btn small danger' data-del='" + (+p.id) + "'>Delete</button>";
      layer.bindPopup(html);
    }
  }).addTo(map);
  overlays["Training areas"] = ann;
  L.control.layers(bases, overlays, { collapsed: true }).addTo(map);
  L.control.scale({ imperial: false }).addTo(map);

  fetch(el.dataset.tilesApi).then(function (r) { return r.json(); }).then(function (d) {
    grid.addData(d.features);
    aoi.addData(d.aoi);
    map.fitBounds(aoi.getBounds(), { padding: [10, 10] });
    window._rockmapTiles = d.features;
  });
  var loadAnn = function () {
    fetch(el.dataset.annApi).then(function (r) { return r.json(); }).then(function (d) { ann.clearLayers(); ann.addData(d); });
  };
  loadAnn();

  var op = document.getElementById("lith-opacity");
  if (op && lith) op.addEventListener("input", function () { lith.setOpacity(op.value / 100); });

  // ------------------------------------------------------------ point query
  var drawing = false;
  map.on("click", function (e) {
    if (drawing) return;
    var url = el.dataset.queryApi + "?lat=" + e.latlng.lat.toFixed(6) + "&lon=" + e.latlng.lng.toFixed(6);
    fetch(url).then(function (r) { return r.json(); }).then(function (q) {
      var html = "<b>" + e.latlng.lat.toFixed(5) + "°N, " + e.latlng.lng.toFixed(5) + "°E</b><br>";
      if (!q.inside) html += "<span class='muted'>outside the region</span>";
      else {
        if (q.class_name !== undefined) html += (q.color ? "<span class='swatch' style='background:" + esc(q.color) + "'></span>" : "") + esc(q.class_name) +
          (q.confidence ? " <span class='muted'>(" + (+q.confidence) + "% confidence)</span>" : "") + "<br>";
        else html += "<span class='muted'>not classified yet</span><br>";
        if (q.elevation_m !== undefined) html += "Elevation: " + Math.round(q.elevation_m) + " m<br>";
        html += "<span class='muted'>tile " + esc(q.tile) + "</span>";
      }
      L.popup().setLatLng(e.latlng).setContent(html).openOn(map);
    });
  });

  // ------------------------------------------------------------ training-area drawing
  var btn = document.getElementById("draw-btn"), fin = document.getElementById("draw-finish"), cancel = document.getElementById("draw-cancel");
  var hint = document.getElementById("map-hint"), hintText = hint ? hint.innerHTML : "";
  var pts = [], preview = null;
  function reset() {
    drawing = false; pts = []; if (preview) { map.removeLayer(preview); preview = null; }
    if (btn) { btn.hidden = false; fin.hidden = true; cancel.hidden = true; }
    map.doubleClickZoom.enable(); el.style.cursor = ""; if (hint) hint.innerHTML = hintText;
  }
  function redraw() {
    if (preview) map.removeLayer(preview);
    var color = colorOf[document.getElementById("draw-class").value];
    preview = pts.length > 2 ? L.polygon(pts, { color: color, weight: 2, fillOpacity: 0.4 }) : L.polyline(pts, { color: color, weight: 2 });
    preview.addTo(map);
  }
  function finish() {
    if (pts.length < 3) { alert("A training area needs at least 3 points."); return; }
    var ring = pts.map(function (p) { return [+p.lng.toFixed(6), +p.lat.toFixed(6)]; });
    ring.push(ring[0]);
    var body = { class_id: +document.getElementById("draw-class").value, geometry: { type: "Polygon", coordinates: [ring] } };
    fetch(el.dataset.annApi, { method: "POST", headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf }, body: JSON.stringify(body) })
      .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw new Error(d.error || r.status); return d; }); })
      .then(function () { reset(); loadAnn(); })
      .catch(function (err) { alert("Could not save: " + err.message); });
  }
  if (btn) {
    btn.addEventListener("click", function () {
      drawing = true; pts = []; btn.hidden = true; fin.hidden = false; cancel.hidden = false;
      map.doubleClickZoom.disable(); el.style.cursor = "crosshair";
      if (hint) hint.innerHTML = "<b>Drawing:</b> click to add points around an area you know is <b>" +
        document.getElementById("draw-class").selectedOptions[0].text + "</b>; double-click or press Finish to save. Only draw where you are confident (field visit, published map).";
    });
    fin.addEventListener("click", finish);
    cancel.addEventListener("click", reset);
    map.on("click", function (e) { if (drawing) { pts.push(e.latlng); redraw(); } });
    map.on("dblclick", function () { if (drawing) finish(); });
    document.getElementById("draw-class").addEventListener("change", function () { if (drawing && pts.length) redraw(); });
  }
  map.on("popupopen", function (e) {
    var b = e.popup.getElement().querySelector("[data-del]");
    if (!b) return;
    b.addEventListener("click", function () {
      if (!confirm("Delete this training area?")) return;
      fetch(el.dataset.annApi + "/" + b.dataset.del, { method: "DELETE", headers: { "X-CSRF-Token": csrf } })
        .then(function (r) { return r.json().then(function (d) { if (!r.ok) throw new Error(d.error || r.status); }); })
        .then(function () { map.closePopup(); loadAnn(); })
        .catch(function (err) { alert(err.message); });
    });
  });

  // ------------------------------------------------------------ "only tiles in view" scope
  var scope = document.getElementById("scope-view");
  document.querySelectorAll(".stage-form").forEach(function (form) {
    form.addEventListener("submit", function () {
      var input = form.querySelector(".tiles-input");
      if (!input) return;
      input.value = "";
      if (scope && scope.checked && window._rockmapTiles) {
        var view = map.getBounds();
        input.value = window._rockmapTiles.filter(function (f) {
          return view.intersects(L.geoJSON(f).getBounds());
        }).map(function (f) { return f.properties.key; }).join(",");
      }
    });
  });

  // ------------------------------------------------------------ live job progress
  document.querySelectorAll(".job-live").forEach(function (box) {
    var bar = box.querySelector(".progress > span"), msg = box.querySelector(".msg");
    var poll = function () {
      fetch(box.dataset.api).then(function (r) { return r.json(); }).then(function (j) {
        bar.style.width = Math.round(j.progress * 100) + "%";
        msg.textContent = j.message || j.status;
        if (["done", "failed", "cancelled"].indexOf(j.status) >= 0) window.location.reload();
        else setTimeout(poll, 2000);
      }).catch(function () { setTimeout(poll, 5000); });
    };
    setTimeout(poll, 1500);
  });
  // refresh the tile grid colours while jobs run
  if (document.querySelector(".job-live")) {
    setInterval(function () {
      fetch(el.dataset.tilesApi).then(function (r) { return r.json(); }).then(function (d) {
        grid.clearLayers(); grid.addData(d.features); window._rockmapTiles = d.features;
      });
    }, 15000);
  }
})();
