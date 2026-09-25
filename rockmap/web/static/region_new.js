/* New-region page: preview preset areas, draw a rectangle, estimate size. */
(function () {
  "use strict";
  var el = document.getElementById("new-map");
  if (!el || !window.L) return;
  var map = L.map(el).setView([35.8, 74.8], 7);
  L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
    { maxZoom: 18, attribution: "&copy; OpenStreetMap contributors" }).addTo(map);
  var shape = null, first = null, mode = "preset";
  var info = document.getElementById("aoi-info");
  var modeInput = document.getElementById("aoi_mode");

  function areaKm2(ring) { // spherical excess approximation, good enough for a size hint
    var R = 6371, a = 0;
    for (var i = 0; i < ring.length - 1; i++) {
      var p1 = ring[i], p2 = ring[i + 1];
      a += (p2[0] - p1[0]) * Math.PI / 180 * (2 + Math.sin(p1[1] * Math.PI / 180) + Math.sin(p2[1] * Math.PI / 180));
    }
    return Math.abs(a * R * R / 2);
  }
  function show(geom) {
    if (shape) map.removeLayer(shape);
    shape = L.geoJSON(geom, { style: { color: "#b5562b", weight: 2, fillOpacity: 0.08 } }).addTo(map);
    map.fitBounds(shape.getBounds(), { padding: [20, 20] });
    var ring = geom.type === "Polygon" ? geom.coordinates[0] : geom.coordinates[0][0];
    var km2 = areaKm2(ring);
    var res = +document.getElementById("resolution").value || 20, ts = +document.getElementById("tile_size").value || 1024;
    var tileKm2 = Math.pow(res * ts / 1000, 2);
    info.textContent = "≈ " + Math.round(km2).toLocaleString() + " km² · roughly " + Math.max(1, Math.round(km2 / tileKm2 * 1.2)) + " tiles of " + (res * ts / 1000).toFixed(1) + " km";
  }
  var sel = document.getElementById("preset");
  function showPreset() { show(JSON.parse(sel.selectedOptions[0].dataset.geom)); }
  sel.addEventListener("change", showPreset);
  showPreset();

  document.querySelectorAll("[data-mode]").forEach(function (b) {
    b.addEventListener("click", function () {
      mode = b.dataset.mode; modeInput.value = mode; first = null;
      if (mode === "preset") showPreset();
      else if (shape && mode === "draw") { map.removeLayer(shape); shape = null; info.textContent = "Click the first corner"; }
    });
  });
  map.on("click", function (e) {
    if (mode !== "draw") return;
    if (!first) { first = e.latlng; info.textContent = "Click the opposite corner"; return; }
    var w = Math.min(first.lng, e.latlng.lng), eX = Math.max(first.lng, e.latlng.lng);
    var s = Math.min(first.lat, e.latlng.lat), n = Math.max(first.lat, e.latlng.lat);
    var geom = { type: "Polygon", coordinates: [[[w, s], [eX, s], [eX, n], [w, n], [w, s]]] };
    document.getElementById("aoi_geojson").value = JSON.stringify(geom);
    first = null;
    show(geom);
  });
  var src = document.getElementById("source"), local = document.getElementById("local-opts");
  if (src && local) src.addEventListener("change", function () { local.hidden = src.value !== "local"; });
})();
