# How to run RockMap

## What you need
* A computer with **Windows 10/11, Linux or macOS**, at least 8 GB RAM (16 GB recommended) and 5 GB of free disk.
* **Python 3.9–3.12** from https://www.python.org/downloads/. On Windows, tick **"Add python.exe to PATH"** during installation.
* An internet connection for the first install and for downloading satellite imagery. The demo itself runs offline.

## Method 1: one command (recommended)

**Windows:** double-click `run.bat`, or run it in a Command Prompt from the project folder:
```bat
run.bat
```
**Linux / macOS:**
```bash
./run.sh
```

The first run creates a virtual environment in `.venv`, installs everything (about 5 minutes) and builds a complete demo (about 1–2 minutes). It then opens the dashboard at **http://127.0.0.1:5000**.

Sign in with **admin / rockmap-demo**. You will be asked to choose your own password straight away.

To check that everything is installed correctly at any time, run `rockmap doctor` (Windows: `.venv\Scripts\rockmap doctor`). Every line should say `OK`.

| Command | What it does |
|---|---|
| `run.bat` / `./run.sh` | Install if needed, build the demo if needed, start the dashboard |
| `run.bat --real` / `./run.sh --real` | Also download and analyse a **real 40 × 40 km area around Gilgit city** from Sentinel-2 (about 2–3 min, needs internet) |
| `run.bat serve` / `./run.sh serve` | Start the dashboard on the existing data |

Stop the server with **Ctrl + C**. Your data is kept in the `data/` folder.

## Method 2: manual installation
```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate        Linux/macOS:  source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu     # CPU build (smaller); omit for GPU
pip install -e ".[dev]"
rockmap quickstart            # build the demo and open the dashboard
# later:
rockmap serve --open          # start the dashboard
pytest                        # run the automated tests
rockmap doctor                # installation self-check
rockmap backup                # back up all data to a zip
```

## Method 3: Docker (servers)
```bash
docker compose up -d --build
docker compose logs web | grep "First start"     # prints the admin password
```
Open http://SERVER:5000. See [DEPLOYMENT.md](DEPLOYMENT.md) for HTTPS, users and backups.

## What the demo contains
1. **Demo: synthetic Karakoram valley** is a fully processed region: cloud-free imagery, surface cover, CNN/RF/SVM models trained from a "digitised geological map" covering the western 60 % of the area, a lithology map for the whole area, mineral-alteration analysis, landslide susceptibility, 8 spectral units, 2 districts with statistics, 30 field observations used for validation, a PDF report and a public share link.
2. With `--real`, **Gilgit & surroundings (real Sentinel-2, 40 × 40 km)**: real satellite composite, snow/water/vegetation cover, mineral-alteration targets, landslide susceptibility and spectral units. No geology is bundled, so name the units yourself (step 3b) to get a rock map.

## Using it for Gilgit-Baltistan
1. **Regions → New region →** choose a preset (Gilgit, Hunza, Skardu, Astore, Chilas, Khaplu) or *Gilgit-Baltistan (whole region)*.
2. On the region page, open **Run everything → Run pipeline**. This runs acquire, analytics and products. It takes about 30–60 s per 20 km tile.
3. Add geology knowledge in any combination:
   * **Name the spectral units** (step 3b): the fastest way to a rock map.
   * **Upload a digitised geological map** (GeoJSON) and/or **draw training areas**, then **Train model** and **Classify**. This is the most accurate option.
   * **Collect field points** with the phone app.
4. **Build products** to get the PDF report, GeoTIFFs with QGIS styles, GeoJSON polygons, target lists and statistics per district.

Details: [GILGIT_BALTISTAN.md](GILGIT_BALTISTAN.md).

## Using the field app on a phone
1. Start the server so other devices can reach it: `rockmap serve --host 0.0.0.0` (or `run.bat serve`, then allow Python through the firewall).
2. On a phone on the same Wi-Fi, open `http://<computer-IP>:5000/field/` (find the IP with `ipconfig` or `ip addr`), then sign in.
3. **GPS needs HTTPS.** Over plain http, tap the map to set the position instead. For GPS in the field, serve RockMap over HTTPS ([DEPLOYMENT.md §3](DEPLOYMENT.md)), or use a tunnel such as `cloudflared tunnel --url http://localhost:5000`.
4. Observations made without a signal are stored on the phone and uploaded automatically when the connection returns. Use *Add to Home screen* to open it like an app.

## Command-line cheat sheet
```bash
rockmap presets                                           # ready-made GB areas
rockmap region create --preset hunza --out regions/hunza
rockmap region acquire regions/hunza                      # Sentinel-2 + DEM
rockmap region analyze regions/hunza --units 10           # targets, hazard, spectral units
rockmap region label-units regions/hunza 1=4 2=7 3=6      # unit=class -> rock map
rockmap region mosaic regions/hunza
rockmap region report regions/hunza --out hunza.pdf
rockmap region query regions/hunza --lon 74.66 --lat 36.32
```

## Troubleshooting
| Problem | Fix |
|---|---|
| `python` not found (Windows) | Reinstall Python with "Add to PATH" ticked, or use `py -3` |
| Torch installation fails | `pip install torch` without the index URL; on Windows use 64-bit Python 3.10–3.12 |
| Port 5000 already used (macOS AirPlay) | `rockmap serve --port 8080 --open` |
| Forgot the admin password | `rockmap create-user admin --reset` |
| Something else is wrong | Run `rockmap doctor`; it names the problem and the fix |
| Downloads fail behind a proxy | Set `HTTPS_PROXY`; if TLS is intercepted, set `CURL_CA_BUNDLE` and `SSL_CERT_FILE` to the proxy CA file |
| Basemap is grey | OpenStreetMap / Esri tiles need internet. The RockMap imagery layers work offline |
