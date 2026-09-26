# Changelog

## 2.2.0: delivery hardening
- **Windows:** data paths are stored portably; downloads and map layers previously failed on Windows. All text files and console output are UTF-8. Tile handles are released and files swapped with retries, so file locking cannot break a rebuild.
- Forced password change for generated, demo or admin-reset passwords; common passwords are rejected.
- `rockmap doctor` (installation self-check), `rockmap backup` / `rockmap restore`, `rockmap create-user --reset`.
- Acceptance tests (complete demo, crawl of every page, ops commands). CI on Linux, Windows and macOS, including the one-command launchers.
- New docs: DELIVERY.md (customer handover checklist).

## 2.1.0: analytics & field
- Mineral-alteration targets, landslide / rockfall susceptibility, spectral units (rock map without training data).
- Offline phone field app with photos; field validation; key-findings summary; public share links; QGIS styles.
- `rockmap quickstart`, `run.sh`, `run.bat`; RUNNING.md, PITCH.md.

## 2.0.0: region platform
- Automatic Sentinel-2 / Copernicus DEM acquisition and cloud-free composites; Gilgit-Baltistan presets; resumable tiled processing; mosaics, statistics, GeoJSON, PDF.
- Accounts and roles, CSRF protection, API tokens, audit log, persistent job queue and worker, map tile server, training-area drawing, Docker Compose.

## 1.0.0: final-year-project system
- Preprocessing, features, CNN / Random Forest / SVM, validation, single-scene dashboard.
