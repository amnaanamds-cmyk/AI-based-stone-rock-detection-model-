/* Field observation form: GPS, tap-to-locate, photo downscaling, offline queue with auto-sync. */
(function () {
  "use strict";
  var form = document.getElementById("obs-form");
  if (!form) return;
  var csrf = document.querySelector('meta[name="csrf-token"]').content;
  var KEY = "rockmap-field-queue";
  var bar = document.getElementById("queue-bar");
  if ("serviceWorker" in navigator) navigator.serviceWorker.register(form.dataset.sw, { scope: "/field/" }).catch(function () {});

  var queue = function () { try { return JSON.parse(localStorage.getItem(KEY) || "[]"); } catch (e) { return []; } };
  var saveQueue = function (q) { try { localStorage.setItem(KEY, JSON.stringify(q)); } catch (e) { alert("Phone storage is full - sync before adding more."); } showQueue(); };
  function showQueue(msg) {
    var n = queue().length;
    bar.hidden = !n && !msg;
    bar.className = "flash " + (n ? "error" : "ok");
    bar.textContent = msg || (n + " observation(s) waiting to upload - they will be sent automatically when online.");
  }

  // map for tap-to-locate
  var map = null, marker = null;
  if (window.L) {
    map = L.map("field-map").setView([35.9, 74.4], 7);
    L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", { maxZoom: 18, attribution: "&copy; OpenStreetMap" }).addTo(map);
    map.on("click", function (e) { setPos(e.latlng.lat, e.latlng.lng, null); });
  }
  function setPos(lat, lon, acc) {
    form.lat.value = lat.toFixed(6); form.lon.value = lon.toFixed(6);
    form.accuracy_m.value = acc === null ? "" : Math.round(acc);
    document.getElementById("gps-info").textContent = acc === null ? "position set from map" : "GPS ±" + Math.round(acc) + " m";
    if (map) { if (marker) marker.setLatLng([lat, lon]); else marker = L.marker([lat, lon]).addTo(map); map.setView([lat, lon], Math.max(map.getZoom(), 13)); }
  }
  document.getElementById("gps-btn").addEventListener("click", function () {
    if (!navigator.geolocation) { alert("GPS not available - tap the map instead."); return; }
    document.getElementById("gps-info").textContent = "locating…";
    navigator.geolocation.getCurrentPosition(function (p) { setPos(p.coords.latitude, p.coords.longitude, p.coords.accuracy); },
      function (err) { document.getElementById("gps-info").textContent = "GPS failed (" + err.message + ") - tap the map. GPS needs HTTPS."; },
      { enableHighAccuracy: true, timeout: 20000, maximumAge: 0 });
  });

  function downscale(file) {
    return new Promise(function (resolve) {
      if (!file) return resolve(null);
      var img = new Image(), url = URL.createObjectURL(file);
      img.onload = function () {
        var s = Math.min(1, 1600 / Math.max(img.width, img.height));
        var c = document.createElement("canvas"); c.width = Math.round(img.width * s); c.height = Math.round(img.height * s);
        c.getContext("2d").drawImage(img, 0, 0, c.width, c.height);
        URL.revokeObjectURL(url); resolve(c.toDataURL("image/jpeg", 0.82));
      };
      img.onerror = function () { resolve(null); };
      img.src = url;
    });
  }
  function send(obs) {
    return fetch(form.dataset.api, { method: "POST", credentials: "same-origin",
      headers: { "Content-Type": "application/json", "X-CSRF-Token": csrf }, body: JSON.stringify(obs) })
      .then(function (r) { if (r.status >= 500 || r.status === 0) throw new Error("server"); return r.json().then(function (d) { if (!r.ok) { d.fatal = true; throw d; } return d; }); });
  }
  function sync() {
    var q = queue();
    if (!q.length || !navigator.onLine) return Promise.resolve();
    var next = q[0];
    return send(next).then(function () { saveQueue(queue().slice(1)); return sync(); })
      .catch(function (e) { if (e && e.fatal) { saveQueue(queue().slice(1)); alert("Rejected: " + e.error); } });
  }
  form.addEventListener("submit", function (ev) {
    ev.preventDefault();
    downscale(form.photo.files[0]).then(function (photo) {
      var obs = { lat: form.lat.value, lon: form.lon.value, accuracy_m: form.accuracy_m.value, class_id: form.class_id.value, gem: form.gem ? form.gem.value : "",
        certainty: (form.querySelector("[name=certainty]:checked") || {}).value || 2, note: form.note.value,
        region_id: form.region_id.value, observed_at: new Date().toISOString(), photo_data: photo };
      var q = queue(); q.push(obs); saveQueue(q);
      form.note.value = ""; form.photo.value = ""; if (form.gem) form.gem.value = "";
      sync().then(function () { showQueue(queue().length ? null : "Saved and uploaded."); });
    });
  });
  window.addEventListener("online", sync);
  showQueue();
  sync();
})();
