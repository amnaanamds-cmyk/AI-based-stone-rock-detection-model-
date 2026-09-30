# Changelog

## 2.6.0: hyperspectral, VHR imagery, GeoPackage
- Hyperspectral mineral mapping (EnMAP, PRISMA, ...) by diagnostic absorption features: alunite, kaolinite, sericite, chlorite, calcite, jarosite, hematite and goethite. Optional Spectral Angle Mapper against a user spectral library. Processing is block-wise, and the copper and iron models use the results where scenes exist.
- Very-high-resolution imagery (WorldView-3, Pleiades, SuperView) as a base map at native resolution (up to zoom 19).
- GeoPackage export of all vector products (pure Python, no GDAL needed), from the dashboard, the CLI and every products run.
- New CLI commands: `region hyperspectral`, `region vhr`, `region geopackage`.

## 2.5.0: structures & mineral prospectivity
- Automatic structural lineaments from the DEM (four-direction hillshade edges plus straightness filter), exported as GeoJSON lines, a density map and a rose diagram.
- Prospectivity models for iron oxide / iron ore, copper alteration (porphyry / vein) and quartz veins (antimony, gold), with ranked target zones as CSV / GeoJSON, map layers and a PDF page.
- Optional ASTER thermal Quartz Index (AST_05 upload or `--aster`).
- Known mineral occurrences (upload, field app, CLI): AUC validation, plus a Random Forest data-driven model with 8 or more occurrences.
- `rockmap region minerals`; pipeline step 1d; the CLI report now includes analytics, gems and minerals.

## 2.4.0: whole-region map
- Imagery is always current: the default season is the last three years including the current one (for example 2024–2026), and among similarly clear scenes the newest are used first.
- Whole-Gilgit-Baltistan overview at 100 m (`run.sh --gb` / `run.bat --gb`, preset *Gilgit-Baltistan overview map*): a complete real map in under an hour.
- Composites now cover every Sentinel-2 granule of large tiles (candidates round-robin per granule) and measure coverage inside the area of interest. Memory is bounded: a scene cap per tile and float16 storage.
- docs/SATELLITE_DATA.md: where the imagery comes from, and why Google Earth imagery cannot be used.

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
