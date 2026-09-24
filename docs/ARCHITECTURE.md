# System design and methodology

## 1. Architecture

```
            ┌──────────────────────── Web dashboard (Flask) ────────────────────────┐
 Browser ◄──┤ pages: dashboard · upload · scene (region select) · job · models      │
 (HTML/CSS/ │ JSON API: /api/jobs/<id> · /api/scenes · /api/models · /api/classes    │
  JS,       │ background job runner (1 worker thread) ── progress ──► SQLite DB     │
  Leaflet)  └───────────────┬───────────────────────────────────────────────────────┘
                            │ calls
 CLI (rockmap …) ───────────┤
                            ▼
      ┌──────────────────── Processing core (rockmap package) ────────────────────┐
      │ io → preprocessing → features → sampling → models (CNN / RF / SVM)        │
      │                                   → pipeline → mapping / evaluation       │
      └───────────────────────────────────────────────────────────────────────────┘
                            │ reads / writes
      data/ ── scenes/<id>/ scene.tif dem.tif reference.tif cloud.tif *.png
            ── models/job<id>/ meta.json cnn.pt rf.joblib svm.joblib cm_*.png
            ── jobs/<id>/ classified.tif confidence.tif *.png result.json
            ── rockmap.db
```

The dashboard and the CLI call the same functions (`train_models`, `classify_scene`), so a result produced in one is identical to the same run in the other.

### Database (SQLite, `data/rockmap.db`)

| Table | Key columns |
|---|---|
| `scenes` | id, name, source (upload/demo), sensor, dos, folder, has_dem, has_reference, has_cloud, width, height, crs, bounds (WGS84 JSON), pixel_size |
| `models` | id, name, folder, scene_id, uses_dem, best_algo, summary (JSON: OA / kappa / F1 per algorithm) |
| `jobs` | id, kind (train/classify), status (queued/running/done/failed), progress, message, params (JSON), scene_id, model_id, folder, error |

## 2. Methodology

### 2.1 Preprocessing
1. **Radiometric scaling.** Digital numbers are converted to reflectance with `DN·scale + offset`. For Sentinel-2 L2A with processing baseline ≥ 04.00 the scale is 1e-4 and the offset is -0.1. For Landsat C2 L2 the scale is 2.75e-5 and the offset is -0.2.
2. **Atmospheric correction.** L2A and C2 L2 products are already surface reflectance. For top-of-atmosphere data (L1C), you can switch on Dark Object Subtraction (DOS1): each band's 0.5th-percentile value is taken as path radiance and subtracted.
3. **Cloud / shadow masking.** The system uses the Sentinel-2 SCL classes (0, 1, 3, 8, 9, 10, 11) or the Landsat QA_PIXEL bits 0 to 5. If neither is available, it falls back to the Haze Optimized Transform: `blue − 0.5·red − 0.08 > 0`, combined with brightness and spectral-flatness tests so that bright carbonates are not masked. Masked pixels are left out of training and get the value 255 in the output map.
4. **Band stacking and co-registration.** The six bands are resampled onto one grid (20 m for Sentinel-2). The DEM and the reference map are reprojected onto the scene grid, the DEM with bilinear resampling and the map with nearest-neighbour.

### 2.2 Features (19 per pixel)
* **Bands:** blue, green, red, NIR, SWIR1, SWIR2.
* **Indices:** NDVI (vegetation), SWIR1/SWIR2 (clay Al-OH and carbonate CO₃ absorptions), red/blue (ferric iron oxides), SWIR1/NIR (ferrous iron, mafic minerals), red/green (iron staining), brightness, and a bare-rock index (SWIR1−NIR)/(SWIR1+NIR).
* **Terrain (from SRTM):** scene-normalised elevation, slope, sin and cos of aspect, hillshade, and roughness (a topographic position index). Rock resistance controls relief, so these features add information that the spectra do not carry.

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

## 3. Limitations and future work
* Reference geological maps are generalised, so the labels near contacts are uncertain. That uncertainty limits how accurate any classifier can appear.
* Vegetation cover hides the rock signal. The project scope targets bare or sparsely vegetated terrain.
* A model trained in one region may not transfer to another, because weathering, illumination and sensor dates differ. For each new region, retrain with local reference data or fine-tune.
* Possible extensions: hyperspectral data (PRISMA, EnMAP), ASTER SWIR/TIR bands, multi-temporal composites, larger-context segmentation networks (U-Net), and active learning using field checks.
