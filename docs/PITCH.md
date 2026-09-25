# RockMap: competition pitch and demo script

## The problem
Gilgit-Baltistan covers about 73,000 km² of some of the world's most rugged terrain. Most of it has only reconnaissance-scale geological maps (1:250,000 to 1:650,000). Field mapping is slow, expensive and dangerous. Meanwhile:
* **Mining and gemstones:** the region hosts gold, copper, marble, and gemstones (ruby, aquamarine, topaz), but exploration targets are identified slowly.
* **Hazards:** landslides and rockfalls repeatedly cut the Karakoram Highway. The 2010 Attabad landslide displaced thousands of people and created a lake that blocked the highway for years.
* **Infrastructure:** CPEC roads, dams and hydropower need rock-type and slope-stability information early in planning.

## The solution
RockMap is a web platform that turns **free satellite data** (Sentinel-2, Copernicus DEM) into **decision-ready geological maps** for any area. A user can go from nothing to a map, a target list and a PDF report in minutes, without GIS or AI expertise.

| For | RockMap delivers |
|---|---|
| Geological Survey / universities | Lithological maps (CNN, RF, SVM), accuracy reports, field validation |
| Mining & exploration companies | Ranked **mineral-alteration targets** (clay, iron-oxide, ferrous) with coordinates, CSV/GeoJSON |
| Disaster management (GBDMA / NDMA), road authorities | **Landslide / rockfall susceptibility** maps and area statistics per district |
| Planners, NGOs, investors | Shareable read-only map links and executive PDF reports |

## What makes it different
1. **Useful without training data.** It produces mineral targets, hazard maps and spectral units immediately. A geologist then names the units to get a rock map in a few clicks. Most AI mapping tools need large labelled datasets first.
2. **Built for high mountains.** It masks snow and glaciers, water, vegetation and shadow, and applies topographic correction to steep slopes.
3. **Honest AI.** The test set is held out by spatial blocks, confidence is mapped for every pixel, maps are validated against field points, and screening results are clearly labelled as such.
4. **End to end.** Data download, processing, AI, a web map, a mobile field app that works offline, reports, a REST API and QGIS styles, all in one open-source package.
5. **Near-zero data cost.** Sentinel-2 (10–20 m, every 5 days) and the Copernicus DEM are free. Commercial imagery costs thousands of dollars per scene.
6. **Scales from one valley to the whole region.** Processing is tiled and resumable. All of GB is about 200 tiles on one workstation.

## Business model
* **SaaS subscription** for survey departments, consultancies and exploration companies (per region / per user).
* **Custom mapping projects:** high-detail maps, field campaigns, training.
* **On-premise licence** for government agencies that need data sovereignty (Docker deployment).

## 7-minute live demo
1. **(30 s) Dashboard:** "One platform, from satellite to decision." Show the regions and jobs.
2. **(90 s) Demo region → Key findings:** read the automatic executive summary. Toggle the **Lithology** layer and click the map: rock type, confidence and elevation.
3. **(60 s) Mineral targets:** open the *Alteration targets* layer, click *zoom* on target #1 and download the CSV. "These are the places a field team should visit first."
4. **(60 s) Landslide susceptibility:** show the hazard layer and the km² per class. "Where the highway and villages are most at risk."
5. **(60 s) Spectral units → rock map:** name two units and click *Create rock map from units*. "No training data needed."
6. **(45 s) Field app on a phone:** record an observation with a photo. It syncs, and appears on the map as validation data.
7. **(45 s) Real Gilgit:** open the real Gilgit region (built with `run.bat --real`). Real Sentinel-2 imagery, snow, rivers and the targets found automatically.
8. **(30 s) Share & report:** open the public share link in a private window and show the PDF report.

## Numbers to quote
* Real Gilgit tile: imagery + DEM acquired in **30–60 s per 20 km tile**. Composites reach **69–100 % cloud- and snow-free** coverage.
* Demo benchmark (synthetic, spatially held-out test): **CNN 98 %** vs Random Forest 92 % and SVM 93 % overall accuracy.
* Field validation on the demo region, in an area the model never saw: about **80 % agreement**.
* 57 automated tests; security features: roles, CSRF protection, login lock-out, audit log, API tokens.

Say clearly in the presentation that the demo accuracy comes from synthetic data. Real accuracy depends on the reference geology used for training.
