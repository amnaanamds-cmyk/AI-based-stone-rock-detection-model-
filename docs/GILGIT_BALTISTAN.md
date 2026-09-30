# Mapping Gilgit-Baltistan with RockMap

This guide takes you from an empty installation to a lithological map of Gilgit-Baltistan (GB).

## 1. Plan

| Item | Whole GB (preset `gilgit-baltistan`) | One valley preset (40 × 40 km) |
|---|---|---|
| Area / tiles | ~72,000 km², ~190 tiles of 20.48 km | ~1,600 km², 9 tiles |
| Imagery downloaded | ~15–25 GB (only the needed parts of each scene are read) | ~1 GB |
| Disk used by the region | ~4–5 GB (composites, DEM, maps, mosaics) | ~150 MB |
| Acquisition time* | 2–4 hours | 5–10 minutes |
| Classification time* (CNN, CPU) | ~1 hour | ~2 minutes |

\* On a 4-core workstation with a good internet connection. Acquisition is limited by download speed. All stages resume where they stopped if interrupted.

**Recommended approach:** start with one valley where you have good reference data (for example Gilgit, Hunza or Skardu). Train and check the model there, then run the whole region with that model.

## 2. Reference (training) data: the part only you can provide

RockMap learns what each rock type looks like from examples. Good sources for GB:
* **Searle, M.P. & Khan, M.A. (1996). Geological Map of North Pakistan and adjacent areas of northern Ladakh and western Tibet, 1:650,000.** This is the standard regional map of the Karakoram, Kohistan and Nanga Parbat.
* Geological Survey of Pakistan (GSP) quadrangle maps.
* Published theses and papers with detailed maps of individual valleys.
* Your own field observations: GPS points and photos of outcrops.

How to prepare them:
1. Georeference the scanned map in QGIS (*Raster ▸ Georeferencer*) and digitise the units as polygons with a `UNIT` attribute. You do not need to digitise everything; a few well-mapped valleys are enough to start.
2. Export as **GeoJSON (EPSG:4326)**.
3. Copy `examples/gb_geology_mapping.json` and edit it so that every unit name in your legend maps to a RockMap class (1–7). Ask a geologist to check the table.
4. In the dashboard, open the region, go to *2 · Training data & model*, upload the GeoJSON, set the attribute (`UNIT`) and paste the mapping.

You can also **draw training areas** directly on the map. Choose a class, click *Draw training area*, click around the area and double-click to finish. Only draw where you are sure of the rock type, and draw several areas per class in different valleys so the model sees natural variation.

Tips for good results:
* Give every class examples from several places, with different slopes, aspects and valleys.
* Avoid drawing over snow patches, river beds, villages or forest. They are masked anyway, but they waste effort.
* Moraine, scree and river terraces belong to class 7 (Quaternary deposits).
* If two classes are always confused (see the confusion matrix), merge them or add more examples.

## 3. Run

| Step | Dashboard | Command line |
|---|---|---|
| Create region | *Regions ▸ New region ▸ Gilgit-Baltistan* | `rockmap region create --preset gilgit-baltistan --out regions/gb` |
| Acquire | *1 · Acquire imagery & DEM* | `rockmap region acquire regions/gb` |
| Train | *2 · Training data & model ▸ Train model* | `rockmap region train regions/gb --reference geology.geojson --field UNIT --mapping mapping.json --out models/gb` |
| Classify | *3 · Classify* | `rockmap region classify regions/gb --model models/gb` |
| Products | *4 · Build products* | `rockmap region mosaic regions/gb` · `stats` · `export` · `report` |

The season defaults to **July–October of the last three years** (for example 2024–2026 in 2026), when seasonal snow cover is smallest. Up to 6 of the least-cloudy scenes per tile are combined. Permanent snowfields and glaciers are mapped as *Snow / Glacier / Ice* instead of being guessed.

## 4. Check the results

* **Accuracy report** (model page, and the PDF report): overall accuracy, kappa and per-class F1 on spatial test blocks that were never used for training. Low recall for a class means it is often missed. Low precision means other rocks are labelled as that class.
* **Confidence layer**: areas with low confidence (red) need more training examples or a field check.
* **Point query**: click the map to compare the prediction with what you know on the ground.
* **Field validation**: take the GeoJSON polygons into the field (for example with QField) and record agreement or disagreement.

## 5. District statistics

Upload the GB district boundaries (GeoJSON with a name attribute) under *4 · Build products*, then build the products. The statistics table, the CSV and the PDF report then contain the area of every rock type per district. District boundaries are not bundled because they change (GB has had several new districts in recent years). Use the current official boundaries.

## 6. Limitations to state in any report

* Only the surface is mapped: no subsurface information, no depth and no mineral grades.
* A 20 m pixel is a mixture of rock, soil, debris and lichen. Thin units narrower than about 60 m are not resolved.
* Debris-covered glaciers can be classified as Quaternary deposits, and very steep north faces may stay in shadow.
* Accuracy depends on the reference map. Generalised 1:650,000 maps limit how detailed and accurate the result can be.
