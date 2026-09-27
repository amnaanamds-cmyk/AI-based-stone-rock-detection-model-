# Changelog

## 2.3.0: gemstone prospectivity
- New gem module: prospectivity models for marble-hosted ruby & spinel, pegmatite gems (aquamarine, topaz, tourmaline, garnet), emerald / beryl contacts, and ultramafic peridot / nephrite. Evidence statistics come from bedrock only; flat ground, fields and snow margins are masked.
- Ranked target zones (top 1 % per setting, at least 500 m apart) with coordinates, area and elevation, as CSV / GeoJSON, a map, the PDF report and share links.
- Known gem localities (CSV / GeoJSON upload, field-app "gem found", CLI): per-model AUC and capture rates, plus a Random Forest presence/background model once 8 or more localities are available.
- New map layers (all settings, each setting, data-driven), `rockmap region gems`, and presets `hunza-gems` and `shigar`.
- Verified on real Sentinel-2 data for Gilgit, Hunza and Shigar. See docs/GEMSTONES.md.

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
