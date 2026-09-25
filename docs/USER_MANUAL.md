# RockMap user manual

> For region-scale mapping (for example all of Gilgit-Baltistan), see **[GILGIT_BALTISTAN.md](GILGIT_BALTISTAN.md)**. Server setup, users and the REST API are covered in **[DEPLOYMENT.md](DEPLOYMENT.md)**. This manual covers installation, the single-scene workflow and the command line.

## 0. Signing in
The first start creates the user `admin`. Its password is printed in the console and saved in `data/initial_admin_password.txt`, unless `ROCKMAP_ADMIN_PASSWORD` was set. Administrators add other users under **Admin ▸ Users**. Roles:
* **viewer** can view maps and download products.
* **analyst** can also upload data, draw training areas and run jobs.
* **admin** can also manage users and delete data.

## 1. Installation
1. Install Python 3.9 or newer.
2. Create a virtual environment: `python -m venv .venv`, then activate it (`source .venv/bin/activate` on Linux/macOS, `.venv\Scripts\activate` on Windows).
3. Optional, to get a smaller download: `pip install torch --index-url https://download.pytorch.org/whl/cpu`
4. Run `pip install -e ".[dev]"` from the project folder.
5. Check the installation with `pytest`. All tests should pass.

Runtime data (uploaded scenes, models, maps and the database) goes to `./data`. To use another folder, set the `ROCKMAP_DATA_DIR` environment variable or pass `rockmap serve --data <folder>`.

## 2. Getting satellite data
| Data | Where to get it | Product to choose |
|---|---|---|
| Sentinel-2 | https://dataspace.copernicus.eu | **L2A** (surface reflectance), cloud cover < 10 %, dry season |
| Landsat 8/9 | https://earthexplorer.usgs.gov | **Collection 2 Level-2** surface reflectance |
| DEM | https://earthexplorer.usgs.gov | SRTM 1 Arc-Second Global |
| Geological map | Geological Survey of Pakistan, or your national survey | Scanned maps must first be digitised in QGIS (polygons with a unit attribute) |

Build the six-band scene. Unzip the Sentinel-2 product, then run:
```
rockmap stack --safe <folder>.SAFE --resolution 20 --scl --out scene.tif
```
For Landsat, or for bands you have exported yourself, list them in the order blue, green, red, NIR, SWIR1, SWIR2:
```
rockmap stack --bands B2.tif B3.tif B4.tif B5.tif B6.tif B7.tif --qa QA_PIXEL.tif --out scene.tif
```
If the scene is large, crop it to your study area first, for example with QGIS *Raster ▸ Extraction ▸ Clip*, or use the region selector in the dashboard.

## 3. Preparing the reference geological map
1. In QGIS, digitise or load the geological map polygons. Each polygon needs an attribute that names its unit, such as `UNIT`.
2. Export the layer as **GeoJSON**.
3. Write a mapping file that assigns each unit to a RockMap class id:
   ```json
   {"Kohat Limestone": 1, "Chorgali Formation": 1, "Murree Formation": 2, "Kuldana Formation": 3,
    "Granite gneiss": 6, "Alluvium": 7}
   ```
4. Rasterise the map onto the scene grid:
   ```
   rockmap rasterize-map --vector geology.geojson --field UNIT --mapping mapping.json --scene scene.tif --out reference.tif
   ```
   Units that are not in the mapping stay unlabelled. The command prints a warning listing them.

The dashboard upload form can do step 4 for you: upload the GeoJSON and paste the mapping into the form.

## 4. Using the dashboard
Start the server with `rockmap serve` and open http://127.0.0.1:5000.

1. **Add a scene.** Use **Upload scene** to add the scene GeoTIFF. You can also add a DEM, a cloud layer and a reference map, and you must choose the sensor preset. Alternatively, click **+ Synthetic study area** on the dashboard to try the system with generated data.
2. **Train.** On a scene that has a reference map, choose the algorithms, the number of CNN epochs and the samples per class, then click **Start training**. A progress bar shows the current CNN epoch. When training finishes, the page shows a benchmark table, per-class metrics, confusion matrices, the CNN training curve and the Random Forest feature importance.
3. **Select a region.** On the scene page, drag a rectangle over the preview to classify only that area. Click **Clear selection** to classify the whole scene. The buttons above the preview switch it between true colour, SWIR false colour, DEM hillshade and the reference map.
4. **Classify.** Pick a model, an algorithm (the default is the best one) and a post-processing filter, then click **Classify**.
5. **Explore the result.**
   * **Map viewer.** The lithology map is drawn over the imagery. The opacity slider controls how much imagery shows through, and the base layer can be switched between true colour, false colour, reference map and confidence.
   * **Web map.** The map is placed on OpenStreetMap or Esri satellite tiles. This needs an internet connection.
   * **Legend & area statistics.** The area of each lithology in km² and as a percentage.
   * **Validation.** Shown when the scene has a reference map: accuracy, kappa, per-class metrics, a confusion matrix and an agreement map.
   * **Downloads.** The map and confidence GeoTIFFs, which open in QGIS with their colours; a PNG map figure for the report; and a JSON report.

## 5. Command-line reference
| Command | Purpose |
|---|---|
| `rockmap demo` | Runs the whole pipeline on a synthetic study area |
| `rockmap synth --out DIR` | Only generates a synthetic scene, DEM and reference map |
| `rockmap stack` | Stacks band files or a Sentinel-2 .SAFE folder into one GeoTIFF |
| `rockmap align-dem --dem D --ref SCENE --out OUT` | Reprojects a DEM onto the scene grid |
| `rockmap rasterize-map` | Turns a vector geological map into a label raster |
| `rockmap train` | Trains the CNN, RF and SVM and writes a model folder with metrics |
| `rockmap classify` | Classifies a scene or a region of it (`--window COL ROW W H`), optionally with `--reference` for validation |
| `rockmap evaluate --pred MAP --ref REF` | Validates any classified map against a reference map |
| `rockmap serve` | Starts the dashboard (production server; `--debug` for development) |
| `rockmap worker` | Runs background jobs in a separate process |
| `rockmap create-user NAME --role analyst` | Creates a dashboard user |
| `rockmap presets` | Lists the ready-made Gilgit-Baltistan areas |
| `rockmap region create/acquire/train/classify/mosaic/stats/export/report/query/status` | Region-scale processing (see GILGIT_BALTISTAN.md) |

Important options:
* `--sensor sentinel2 | sentinel2_legacy | landsat89 | reflectance` sets how DN values are scaled.
* `--dos` applies dark object subtraction, for L1C or TOA data.
* `--samples`, `--epochs`, `--block` set samples per class, CNN epochs and the spatial block size.
* `--smoothing 1 | 3 | 5` sets the majority filter size (1 = off).

## 6. Troubleshooting
| Problem | Fix |
|---|---|
| "The scene has N bands; 6 are required" | Build the scene with `rockmap stack` so it has the correct band order |
| "This model was trained with DEM terrain features" | Upload a DEM for the scene, or train a model without the DEM option |
| "Too few labelled training pixels" | The reference map does not overlap the scene, or the mapping file left every unit unmapped |
| Clouds classified as limestone | Provide the SCL or QA_PIXEL layer, or choose a clearer scene |
| Web map is blank | It needs internet access for Leaflet and the basemap tiles; use the Map viewer offline |
| Training is slow | Lower `--samples` or `--epochs`. The CNN uses a GPU automatically if one is available |
