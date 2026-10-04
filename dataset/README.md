# Dataset: where your training data goes

This project classifies **rock types from satellite images**. A training example is not a photo with a label, but **an image of an area plus a map that says which rock is where**. The model learns from every labelled pixel.

## 1. Folder structure

Put each study area in its own folder inside `dataset/raw/`:

```
dataset/raw/
├── gilgit_2025/              <- any folder name (letters, numbers, _ and -)
│   ├── image.tif             REQUIRED  satellite image
│   ├── labels.tif            REQUIRED  reference map (or labels.geojson + area.json)
│   ├── dem.tif               optional  elevation model
│   └── area.json             optional  settings for this area
├── hunza_field_2026/
│   ├── image.tif
│   └── labels.geojson
└── ...
```

All areas in `dataset/raw/` are used together for training. To add data later, add another folder and train again.

## 2. The files

### image.tif (required)
- A **GeoTIFF** (it must have a coordinate system) with **6 bands in this order: blue, green, red, near-infrared, SWIR-1, SWIR-2**.
  - Sentinel-2: B2, B3, B4, B8, B11, B12, resampled to 20 m.
  - Landsat 8/9: SR_B2 … SR_B7.
- Pixel values can be any of these:

| Pixel values | What to do |
|---|---|
| Reflectance 0–1 (float) | nothing, detected automatically |
| Integers 0–10000 (reflectance × 10000, e.g. RockMap region tiles `data/regions/<id>/tiles/*/stack.tif`, most exported products) | nothing, detected automatically |
| Raw Sentinel-2 L2A bands, processing baseline 04.00 or later (with the −1000 offset) | add `"sensor": "sentinel2"` to `area.json` |
| Landsat 8/9 Collection 2 Level-2 | add `"sensor": "landsat89"` to `area.json` |

- **Easiest way to get images:** create a region in the RockMap dashboard and acquire imagery. Then copy a tile's `stack.tif` as `image.tif` and its `dem.tif` as `dem.tif`. To stack your own Sentinel-2 download, use `rockmap stack --safe <folder>.SAFE --resolution 20 --out image.tif`.

### labels.tif (required, option A): a label raster
- One band. Each pixel holds a **class id** (table below), and **0 means unlabelled**: unknown, or not sure.
- It can have a different resolution or projection from `image.tif`; it is aligned automatically. It only has to overlap the image.
- You only need to label the parts you are sure about. Unlabelled pixels (0) are ignored.

### labels.geojson (required, option B): geological map polygons
- Polygons digitised from a geological map, e.g. in QGIS. Any coordinate system works.
- Tell the system which attribute holds the rock unit, and how units translate to classes, in `area.json`:

```json
{
  "label_field": "UNIT",
  "mapping": {
    "Kohat Limestone": 1,
    "Murree Formation": "Sandstone",
    "Chilas Complex": "Basalt / Mafic Igneous",
    "Qa": 7
  }
}
```

  Mapping values can be class ids or class names. Units not listed in `mapping` stay unlabelled. If the field already contains class ids, `mapping` can be left out. A template is in `examples/area.json`.

### dem.tif (optional)
An elevation model (e.g. Copernicus DEM or SRTM), in any resolution or projection. With a DEM, five terrain features (slope, aspect, relief, …) are added. **Either every area has one, or none** (`USE_DEM = "auto"` in `training/config.py` handles this). A model trained with a DEM needs a DEM again at prediction time.

### area.json (optional)
Settings for one area: `sensor`, `label_field`, `mapping`, `description`.

## 3. Classes

| id | Class | Examples |
|---|---|---|
| 1 | Limestone / Carbonate | limestone, dolomite, marble |
| 2 | Sandstone | sandstone, quartzite |
| 3 | Shale / Mudstone | shale, mudstone, slate |
| 4 | Granite / Felsic Igneous | granite, granodiorite (Karakoram and Kohistan batholiths) |
| 5 | Basalt / Mafic Igneous | basalt, gabbro, ophiolite (Chilas, Kohistan arc) |
| 6 | Metamorphic (Schist / Gneiss) | schist, gneiss, phyllite |
| 7 | Alluvium / Quaternary Deposits | river terraces, moraine, scree |
| 0 | unlabelled | ignored in training |

Snow, water, dense vegetation, deep shadow and cloud are masked automatically and are never used as rock samples. You don't need to label them.

The classes are defined once for the whole system in `rockmap/config.py` (`ROCK_CLASSES`). To add a class, see `training/README.md`.

## 4. How much data?

| | Minimum | Recommended |
|---|---|---|
| Classes | 2 | all classes that occur in your area |
| Labelled pixels per class | about 500 (≈ 0.2 km² at 20 m) | 5,000 or more, spread over several places |
| Areas | 1 | 2 or more, so you can test on an area the model never saw |

Spread the labels: many small patches in different valleys are worth more than one big polygon. The test set is taken from separate spatial blocks, so labels concentrated in one spot give few test pixels.

## 5. Check before training

```bash
python training/train.py --check
```

This lists the areas found and every problem: wrong band count, no coordinate system, labels that don't overlap, unknown class ids, unmapped units, and so on.

## 6. Adding more data later

1. Add a new area folder, or improve the labels of an existing one.
2. Run `python training/train.py`.

All areas are processed again (unchanged areas come from the cache in `dataset/processed/`), and a **new model version** is saved. The old versions are kept.

## 7. Sample data (only for trying the pipeline)

```bash
python training/make_sample_dataset.py     # creates dataset/raw/sample_synthetic/
```

This is **synthetic** data: simulated rock units, spectra and terrain. Models trained only on it are flagged *SAMPLE DATA* in reports and in the application. Delete `dataset/raw/sample_synthetic/` when you have real data.

`dataset/raw/` and `dataset/processed/` are not committed to git. Keep your own backup of your dataset.
