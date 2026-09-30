# Structures and mineral prospectivity

RockMap turns free satellite data into the first-pass products of a commercial mineral exploration programme:

- **structural lineaments** (candidate faults and fractures), with a density map and a rose diagram
- **prospectivity maps** for iron, copper and quartz-vein antimony / gold
- **ranked target zones** as CSV and GeoJSON
- **validation** against known occurrences
- a **data-driven model** once enough occurrences are known

![Gilgit: lineaments and mineral targets](images/minerals_gilgit.jpg)

## 1. What is produced

| Product | Contents | Formats |
|---|---|---|
| Lineaments | Straight topographic features extracted from the Copernicus DEM, with strike and length | Map layer, `lineaments.geojson`, rose diagram (dashboard and PDF) |
| Lineament density | Lineament length per area (km/km²) in a 2 km window | Used as evidence |
| Iron model | Iron oxide (red/blue) and gossan (SWIR1/red) on fractured ground | Map layer `Minerals - iron oxide / iron ore` |
| Copper model | Clay / sericite (SWIR1/SWIR2) with an iron-oxide cap and high lineament density | Map layer `Minerals - copper alteration` |
| Quartz-vein model | Lineament density and proximity in pale, iron-poor rock, plus the ASTER Quartz Index when imported | Map layer `Minerals - quartz veins (antimony, gold)` |
| Target zones | Top 1 % of the region per model, ranked, at least 500 m apart, at most 50 per model | `mineral_targets.csv`, `.geojson`, PDF page |
| Validation | AUC and top-10 % / top-20 % capture rate of known occurrences | Dashboard, PDF |
| Data-driven model | Random Forest trained on 8 or more known occurrences, cross-validated by spatial group | Map layer `Minerals - data-driven` |

**How to run it:**
- Dashboard: **1d · Lineaments & mineral prospectivity**. It also runs as part of *Run everything*.
- Command line: `rockmap region minerals <region> [--occurrences known.csv] [--aster ast05_*.tif]`.

## 2. What the satellite can and cannot see

| Target | Sentinel-2 + DEM (built in, free) | Needs extra data |
|---|---|---|
| Iron oxide / gossan | ✅ Direct (red/blue, SWIR1/red) | — |
| Clay / sericite alteration (copper) | 🟡 Detects hydroxyl minerals as a group | ✅ Kaolinite vs alunite vs sericite vs chlorite: **hyperspectral import (section 4)** |
| Structures (faults, veins, dykes) | 🟡 Lineaments of 400 m and longer from a 30 m DEM | Metre-scale veins: **very-high-resolution imagery upload (section 5)** and field mapping |
| Quartz | ❌ No feature in visible or SWIR light | **ASTER thermal bands 10–14**: import them to get the Quartz Index |
| Antimony (stibnite) | ❌ No diagnostic spectral signature at any resolution | Only through its quartz-vein host and structural setting, then field sampling |

Every PDF report states these limits. Target zones are **priorities for field checking, not proven deposits.**

## 3. Adding ASTER thermal data (quartz)

1. Register (free) at https://urs.earthdata.nasa.gov and open https://search.earthdata.nasa.gov.
2. Search for **AST_05** (ASTER L2 Surface Emissivity), draw your area, and choose cloud-free, snow-free scenes, preferably July–October.
3. Convert each HDF file to a GeoTIFF with **bands 10, 11 and 12 in that order**, using QGIS or `gdal_translate`.
4. Upload the files in the dashboard (step 1d → *Upload ASTER & recompute*), or run `rockmap region minerals <region> --aster file1.tif file2.tif`.

RockMap computes the Quartz Index (Rockwell & Hofstra 2008), QI = e11² / (e10 · e12), for every tile. The quartz-vein model then gives it 45 % of its weight.

## 4. Hyperspectral mineral mapping (EnMAP, PRISMA)

Sentinel-2 detects "clay" but cannot tell the alteration minerals apart. Hyperspectral sensors (about 10 nm bands from 420 to 2450 nm) can, and that is what zones a porphyry-copper system.

| Mineral | Diagnostic features (nm) | Alteration meaning |
|---|---|---|
| Alunite | 1762, 2165 (stronger than 2205) | Advanced argillic (lithocap, epithermal) |
| Kaolinite | 2165 + 2205 doublet | Argillic |
| Sericite / muscovite / illite | 2200 (single) + 2350 | Phyllic, the core of porphyry systems |
| Chlorite / epidote | 2250 + 2335 | Propylitic |
| Calcite | 2340 (no 2200, no 2250) | Carbonate, skarn host |
| Jarosite | 2265 | Acid-sulfate weathering of sulfides |
| Hematite / goethite | around 870 / 925 | Iron oxide, gossan |

**Method:**
1. Measure continuum-removed band depth at each mineral's main feature, relative to a straight line between the feature's two shoulders.
2. Check secondary features, and features that must be absent, to separate look-alikes. For example, kaolinite needs the 2165 nm shoulder, while sericite must not have it.
3. Optionally, run a **Spectral Angle Mapper** against your own spectral library: a CSV with a `wavelength` column plus one column per mineral, e.g. exported from the public-domain USGS Spectral Library splib07. SAM compares continuum-removed absorption depths, so featureless rock is not matched to anything.

**Where the hyperspectral maps reach, they replace most of the Sentinel-2 evidence:**
- Copper model: 40 % weight on alunite / kaolinite / sericite depth.
- Iron model: 40 % weight on hematite / goethite depth.

**How to get and load the data:**
1. **EnMAP:** free for scientific and commercial use after registration at https://planning.enmap.org (DLR). Order **L2A** (surface reflectance) scenes and export them to GeoTIFF.
2. **PRISMA:** free after registration at https://prisma.asi.it (ASI). Use **L2D** reflectance and convert the HE5 file to GeoTIFF, e.g. in QGIS or with the EnMAP-Box plug-in.
3. Load the scenes in one of these ways:
   - **Dashboard:** step 1d → *Upload hyperspectral & recompute*. Add the ENVI `.hdr` file if the band wavelengths are not stored in the GeoTIFF.
   - **Command line:** `rockmap region hyperspectral <region> scene.tif [--wavelengths scene.hdr] [--library usgs.csv]`.

Scenes are processed block by block and only the needed bands are read, so a full EnMAP scene fits on a laptop.

**Not mappable:** stibnite and quartz have no SWIR features. Quartz needs ASTER thermal data (section 3).

## 5. Very-high-resolution imagery (WorldView-3 and similar)

Purchased ortho-images (WorldView-3, Pleiades, SuperView, GaoJing, …) can be uploaded in step 1d, or with `rockmap region vhr <region> image.tif [--bands R G B]`. They are kept at native resolution (0.3–2 m), stretched to RGB and served as a base map up to zoom level 19. This lets geologists interpret veins, dykes and outcrops directly over the ranked targets. Band order is detected automatically: 8-band WorldView-3 MS uses bands 5, 3, 2 and 4-band images use 3, 2, 1.

## 6. GIS delivery: one GeoPackage

*Downloads → GeoPackage (all vectors)*, or `rockmap region geopackage <region>`, produces one `.gpkg` file that **ArcGIS Pro and QGIS open directly**. It contains:
- area of interest
- lithology polygons
- alteration, mineral and gem targets
- lineaments
- known mineral and gem occurrences
- field observations

It is also written automatically with every *Build products* run. Raster layers are delivered as GeoTIFF mosaics with QGIS styles.

## 7. Known occurrences (validation and the data-driven model)

- **Upload:** in the dashboard, use *Mineral prospectivity & structures → Known mineral occurrences*. Upload a CSV with columns `lat, lon, commodity, name` (template: `examples/mineral_occurrences_template.csv`) or GeoJSON points with a `commodity` property.
- **Recognised commodities:** iron, haematite, magnetite, copper, porphyry, molybdenum, antimony, stibnite, gold, quartz. Other commodities are stored but not validated.
- **Field app:** choose *Mineral occurrence here*. These observations are added automatically.
- **Good sources:** Geological Survey of Pakistan mineral maps (e.g. the *Mineral Map of Gilgit-Baltistan*), published papers, and company records. Use surveyed mine or pit coordinates.

With 8 or more occurrences inside the region, a Random Forest is trained on the six evidence layers and scored with spatially grouped cross-validation (AUC). This matches the "Phase 2 — supervised ML refinement" that exploration consultancies sell separately.

## 8. Method details (for reports)

**Lineaments:**
1. Compute hillshades of the DEM lit from 0°, 45°, 90° and 135° at 30° elevation.
2. Detect edges with a Sobel filter, keeping the top 15 % of each hillshade.
3. Keep a pixel as a lineament when at least 70 % of a 600 m line through it, in one of 12 directions, is also edge.
4. Vectorise connected runs of equal strike into straight segments of at least 400 m.

In high mountains, many lineaments are ridges and valleys rather than faults. They are candidates for a geologist to interpret.

**Evidence:**
1. Convert each band ratio to a robust z-score (median and MAD) over usable bedrock in the whole region. Usable means slope ≥ 15°, and no snow, vegetation, water or data gaps.
2. Map each z-score to a fuzzy membership from 0 to 1 (3 σ = 1).
3. Score each model as the weighted sum of its memberships (0.7 = 100). The vein model is also multiplied by host-rock evidence, because structures alone are everywhere in the Karakoram.

**Targets:** connected zones above each model's own 99th percentile (at least 75), with a minimum of 12 pixels, ranked by mean score × log(size).
