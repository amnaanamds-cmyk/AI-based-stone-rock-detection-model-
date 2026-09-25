# System design and methodology

## 1. Architecture

```
 Browser (HTML/JS, bundled Leaflet)          Scripts / QGIS / other systems
        │ session cookie + CSRF                      │ Bearer API token
        ▼                                            ▼
 ┌──────────────── Web platform (Flask + waitress) ────────────────────────────┐
 │ auth (roles, lock-out, audit) · scenes · regions · models · jobs · admin    │
 │ REST API · XYZ tile server (region mosaics → Web Mercator PNG, disk cache)  │
 └───────────────┬───────────────────────────────┬────────────────────────────┘
                 │ enqueue                        │ read
                 ▼                                ▼
        SQLite (WAL): users, scenes, regions, models, jobs, annotations, audit
                 ▲ claim / progress / heartbeat
 ┌───────────────┴──────── Workers (threads in the server or `rockmap worker`) ┐
 │ scene jobs: train · classify          region jobs: acquire · train ·        │
 │                                        classify · products · pipeline       │
 └───────────────┬─────────────────────────────────────────────────────────────┘
                 ▼
 ┌──────────────────────── Processing core (rockmap package) ─────────────────┐
 │ acquisition: STAC / S3 search by MGRS, COG window reads, composites, DEM    │
 │ region: UTM tile grid, resumable stages, VRT halo reads, mosaics, stats,    │
 │         GeoJSON export, point query          report: PDF                    │
 │ pipeline: preprocessing → masks → features → CNN / RF / SVM → blocks        │
 └────────────────────────────────────────────────────────────────────────────┘
        ▲ HTTPS range requests
 AWS Open Data: sentinel-cogs (Sentinel-2 L2A COG), copernicus-dem-30m (GLO-30)
```

The dashboard, the REST API and the command line all call the same core functions, so a result produced one way is the same as one produced another way.

### Data layout (`ROCKMAP_DATA_DIR`)
```
rockmap.db                       SQLite database
regions/<id>/region.json         configuration (AOI, CRS, resolution, season)
            grid.json            tile grid;  state.json  per-tile status
            tiles/<tx>_<ty>/     stack.tif (6 bands + SCL code, uint16) dem.tif classified.tif confidence.tif meta.json
            stack.vrt dem.vrt    virtual mosaics used for seamless halo reads
            mosaic/*.tif         region-wide tiled GeoTIFFs with overviews
            products/            stats.json lithology.geojson report.pdf
            references/          uploaded geological maps
models/<name>/                   meta.json cnn.pt rf.joblib svm.joblib cm_*.png
scenes/<id>/ · jobs/<id>/        single-scene workflow
logs/job<id>.log                 per-job logs
```

### Database tables
| Table | Purpose |
|---|---|
| `users` | username, password hash, role, API-token hash, active flag |
| `regions` | name, folder, preset, bounds |
| `scenes` | single-scene uploads and their layers |
| `models` | trained bundles with a summary (OA / kappa / F1 per algorithm), source scene or region |
| `jobs` | kind, status, progress, message, params, heartbeat, attempts, cancel flag, user |
| `annotations` | training polygons drawn on the map (class, geometry, author) |
| `audit` | who did what, and when |

### Region processing
1. **Grid.** The AOI is projected to the UTM zone of its centroid (EPSG:32643 for Gilgit-Baltistan) and covered with square tiles, 1024 px × 20 m = 20.48 km by default. Only tiles that touch the AOI are kept.
2. **Acquire (per tile).** The DEM comes from Copernicus GLO-30 1° COGs, reprojected onto the tile. Scenes are searched with the STAC API, or by listing the S3 bucket by MGRS granule when STAC is unreachable. Up to *N* least-cloudy scenes are read. For each scene, the six bands plus SCL are read with an HTTP range request, warped onto the tile, cloud and shadow are removed, and C-correction is applied. The tile value is the per-pixel median of clear observations. Snow observations are used only where no snow-free observation exists. Pixels outside the AOI are set to no-data.
3. **Train.** Tiles that overlap reference polygons or rasters are sampled with the same stratified spatial-block sampler used for single scenes, with a per-tile share of the class quota. The normaliser and all models are then fitted once.
4. **Classify.** Each tile is predicted from `stack.vrt` / `dem.vrt` with a 14-pixel halo, so CNN context, terrain derivatives and the majority filter are identical across tile edges. Class counts per tile are stored for fast statistics.
5. **Products.** Mosaics are written tile by tile into one tiled GeoTIFF per layer, then overviews are built. Statistics are the sum of the tile counts; district statistics come from rasterising the districts on each tile. Polygons come from a sieve followed by polygonisation per tile. The PDF is drawn with matplotlib.

## 2. Methodology

### 2.1 Preprocessing
1. **Radiometric scaling.** Digital numbers are converted to reflectance with `DN·scale + offset`. For Sentinel-2 L2A with processing baseline ≥ 04.00 the scale is 1e-4 and the offset is -0.1. For Landsat C2 L2 the scale is 2.75e-5 and the offset is -0.2.
2. **Atmospheric correction.** L2A and C2 L2 products are already surface reflectance. For top-of-atmosphere data (L1C), you can switch on Dark Object Subtraction (DOS1): each band's 0.5th-percentile value is taken as path radiance and subtracted.
3. **Cloud / shadow masking.** The system uses the Sentinel-2 SCL classes (0, 1, 3, 8, 9, 10, 11) or the Landsat QA_PIXEL bits 0 to 5. If neither is available, it falls back to the Haze Optimized Transform: `blue − 0.5·red − 0.08 > 0`, combined with brightness and spectral-flatness tests so that bright carbonates are not masked. Masked pixels are left out of training and get the value 255 in the output map.
4. **Band stacking and co-registration.** The six bands are resampled onto one grid (20 m for Sentinel-2). The DEM and the reference map are reprojected onto the scene grid, the DEM with bilinear resampling and the map with nearest-neighbour.

### 2.1b Surface masks (mountain terrain)
Before classification, pixels where rock cannot be seen are labelled with their own classes:
* **snow / glacier / ice**: NDSI > 0.4 with green > 0.15 and NIR > 0.11, or SCL 11
* **water**: NDSI > 0.1 with dark NIR, or NDWI > 0.15, or SCL 6
* **dense vegetation**: NDVI > 0.45
* **terrain shadow**: mean reflectance < 0.035, or cos(i) < 0.02 from the DEM and the sun position

Masked pixels are left out of training and statistics, and appear as separate classes on the map.

### 2.2 Features (19 per pixel)
* **Bands:** blue, green, red, NIR, SWIR1, SWIR2.
* **Indices:** NDVI (vegetation), SWIR1/SWIR2 (clay Al-OH and carbonate CO₃ absorptions), red/blue (ferric iron oxides), SWIR1/NIR (ferrous iron, mafic minerals), red/green (iron staining), brightness, and a bare-rock index (SWIR1−NIR)/(SWIR1+NIR).
* **Terrain (from the Copernicus / SRTM DEM):** absolute elevation (km, comparable across tiles), slope, sin and cos of aspect, hillshade, and roughness (a topographic position index). Rock resistance controls relief, so these features add information that the spectra do not carry.

Features are z-score normalised with mean and standard deviation computed on the training pixels. These values are stored with the model and reused unchanged at inference.

### 2.3 Training data
* Labels come from the reference geological map, rasterised onto the scene grid.
* **Spatial block split.** The scene is cut into 32 × 32 px blocks, which are randomly assigned 60 % to training, 15 % to validation and 25 % to testing. Neighbouring pixels are highly correlated, so a pixel-level random split would put near-duplicate pixels in both training and test sets and inflate the accuracy. The block split avoids that.
* **Stratified sampling.** Up to N pixels are drawn per class (default 3,000 to 4,000). Training and validation samples skip pixels on map contacts, where the reference map is least reliable. The test set keeps them.

### 2.4 Models
* **CNN (main model).** A fully convolutional network: a 1×1 spectral-mixing stem, then four unpadded 3×3 convolution + batch-norm + ReLU blocks (64 → 128 channels), then a 1×1 head with dropout. The receptive field is 9 × 9 pixels (180 m at 20 m resolution). It is trained on 9 × 9 patches to predict the centre pixel, using class-weighted cross-entropy with label smoothing, AdamW and a one-cycle learning-rate schedule. Patches are augmented with random flips and 90° rotations. The weights from the epoch with the best validation accuracy are kept. Because the network is fully convolutional, a whole scene is classified in one dense pass (tiled at 512 px) instead of one patch at a time.
* **Random Forest.** 200 trees, `min_samples_leaf=2`, balanced class weights.
* **SVM.** RBF kernel with C = 10 and `gamma="scale"`, balanced class weights, trained on at most 6,000 samples because SVM training time grows quadratically. Its confidence is a softmax over the one-vs-rest decision values.

### 2.5 Map generation
The system produces a per-pixel argmax class and a confidence value (the maximum class probability), and can apply an optional 3×3 or 5×5 majority filter. It writes a GeoTIFF with an embedded colour table, a colour PNG, a map figure with a legend, and the area of each class in km².

### 2.6 Validation
* **Model level.** Held-out spatial test blocks: overall accuracy, Cohen's kappa, macro F1, per-class precision (user's accuracy), recall (producer's accuracy) and a confusion matrix.
* **Map level.** Every valid pixel of the classified map is compared with the reference geological map, and an agreement map (green = agrees, red = disagrees) is produced.

### 2.7 Analytics that need no training data (`rockmap/analytics.py`, `Region.analyze`)
* **Mineral alteration.** For each usable pixel the system computes four ratios: clay/hydroxyl SWIR1/SWIR2, iron oxide red/blue, ferrous SWIR2/NIR + green/red, and gossan SWIR1/red. It then takes a robust z-score of each ratio against the region-wide median and MAD. Only positive anomalies count, and they are combined with weights of 0.40 / 0.35 / 0.15 / 0.10. A combined score of 3 robust sigma maps to 100.

  Several pixel types are excluded because ratios there are unreliable: dark pixels (mean reflectance below 0.08), partial snow (NDSI above 0.15), and a 100 m buffer around snow, cloud and no-data. Vegetation above NDVI 0.25 is excluded for the SWIR ratios only.

  Connected areas scoring 75 or more and covering at least 1 ha become **targets**. Each target records its centroid, area, mean and peak score, dominant anomaly type, elevation and lithology. Targets are ranked by mean score × log(size).
* **Landslide / rockfall susceptibility.** A knowledge-driven weighted overlay of five factors:
  * slope (40 %), with a piecewise response that peaks at 40–50°
  * relief within 300 m (15 %)
  * proximity to rivers (15 %), using exp(−distance/400 m)
  * rock weakness from the lithology map (20 %): shale and Quaternary deposits are weakest, granite strongest
  * absence of vegetation (10 %)

  The index is split into 5 classes at 0.35 / 0.50 / 0.65 / 0.80. Snow and water are excluded.
* **Spectral units.** Mini-batch k-means is run on a region-wide pixel sample. The inputs are brightness-normalised spectra, log brightness and the 7 indices, so illumination affects the result less. Each unit gets a confidence equal to the margin between the nearest and second-nearest cluster centre. A geologist maps units to rock classes, which gives a lithology map with no model training (`Region.label_clusters`).
* **Field validation.** Phone observations are sampled against the current map, and the system reports agreement, kappa and a confusion matrix. Observations marked *probable* or *certain* also become 40 × 40 m training squares.
* **Insights.** Rule-based sentences generated from the statistics and analytics feed the dashboard and the executive-summary page of the PDF report.

## 3. Limitations and future work
* Reference geological maps are generalised, so the labels near contacts are uncertain. That uncertainty limits how accurate any classifier can appear.
* Vegetation cover hides the rock signal. The project scope targets bare or sparsely vegetated terrain.
* A model trained in one region may not transfer to another, because weathering, illumination and sensor dates differ. For each new region, retrain with local reference data or fine-tune.
* Possible extensions: hyperspectral data (PRISMA, EnMAP), ASTER SWIR/TIR bands, multi-temporal composites, larger-context segmentation networks (U-Net), and active learning using field checks.
