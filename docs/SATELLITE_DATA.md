# Where the satellite imagery comes from

**Short answer:** RockMap downloads all imagery itself, for free, from the European Space Agency's **Sentinel-2** satellites and the **Copernicus DEM**. You need no account, no manual downloads and no Google Earth. To get a map of the **whole of Gilgit-Baltistan** in about 40 minutes:

| Windows | Linux / macOS | Dashboard |
|---|---|---|
| `run.bat --gb` | `./run.sh --gb` | **Regions → New region → "Gilgit-Baltistan overview map (whole region at 100 m)" → Run everything** |

This produces a complete, real map of the entire region with these layers:
* true colour
* false colour
* spectral rock units
* mineral-alteration targets
* landslide / rockfall susceptibility
* gemstone prospectivity
* snow, glacier and water

For detailed work, zoom in and create **20 m regions** (the presets *Gilgit*, *Hunza*, *Skardu*, *Shigar*, … or a rectangle you draw) over the areas you care about.

## Why not Google Earth?

Google Earth is excellent to *look at*, but it cannot be used for rock or mineral mapping, and it may not be downloaded:

| | Google Earth / Google Maps imagery | Sentinel-2 (what RockMap uses) |
|---|---|---|
| **Licence** | Bulk download and use as data are **prohibited** by the Google Maps / Earth terms of service. A product built on scraped Google imagery cannot be sold. | **Free and open** for any use, including commercial (Copernicus licence). |
| **Spectral bands** | Only red, green and blue: a photograph | 13 bands, including **near-infrared and two short-wave-infrared (SWIR) bands** |
| **Rock / mineral information** | Colour only. Different rocks with similar colours cannot be told apart. | Clay, carbonate (marble), iron-oxide, mafic and Mg-OH minerals leave fingerprints in SWIR. These fingerprints are what the AI classifier, alteration targets and gem models use. |
| **Calibration** | A mosaic of different dates, satellites and colour-balanced aerial photos | Calibrated surface reflectance, with the same physics everywhere |
| **Updates** | Irregular; many GB areas are years old | A new image every 5 days, and RockMap builds cloud- and snow-free composites from 2023–2025 |
| **Resolution** | Sharper (0.3–15 m) | 10–20 m, enough for rock formations, alteration zones and marble or pegmatite belts |

The dashboard **does** show high-resolution satellite photos as a *background* to help you recognise places. Use the layer switcher (top right of the map) and choose **Esri World Imagery** or **OpenStreetMap**. This background is for display only; the analysis always runs on Sentinel-2.

## How the download works (technical)

* **Sentinel-2 L2A**: surface reflectance, served as Cloud-Optimised GeoTIFFs from the AWS Open Data bucket `sentinel-cogs` (Element 84 Earth Search). RockMap selects the Sentinel-2 grid squares covering each tile and the late-summer months (July–October, the least snow). It reads only the pixels it needs, masks cloud, shadow and snow with the scene classification layer, and computes a median composite.
* **Copernicus DEM GLO-30**: 30 m elevation from the AWS bucket `copernicus-dem-30m`, used for slope, hillshade, topographic correction, landslide susceptibility and the gem models.
* **Size and time** (typical connection):

| Area | Resolution | Tiles | Time | Disk |
|---|---|---|---|---|
| 40 × 40 km valley (preset) | 20 m | 9 | 10–20 min | ~150 MB |
| Whole Gilgit-Baltistan, overview | 100 m | 15 | ~40 min | ~0.5 GB |
| Whole Gilgit-Baltistan, detailed | 20 m | ~190 | several hours (resumable) | ~5–8 GB |

* **Offline or blocked internet?** Download Sentinel-2 L2A `.SAFE` products manually from https://dataspace.copernicus.eu (free account). Then use *Scenes → Upload*, or `rockmap stack --safe <folder>.SAFE --scl --out scene.tif`, and create a region with the source *local scenes*. Landsat 8/9 Collection 2 Level-2 (https://earthexplorer.usgs.gov) works the same way.

## Credits to show on maps and reports
“Contains modified Copernicus Sentinel data [year]” and “Copernicus DEM © DLR e.V. 2010–2014 and © Airbus Defence and Space GmbH 2014–2018, provided under COPERNICUS by the European Union and ESA”.
