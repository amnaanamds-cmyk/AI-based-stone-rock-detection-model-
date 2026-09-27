"""Region-scale processing (e.g. all of Gilgit-Baltistan).

A *region* is an area of interest (AOI) cut into a grid of square processing tiles in a
UTM projection. Every stage is resumable and works tile by tile, so regions of any size
(Gilgit-Baltistan is ~73,000 km2, about 200 tiles of 20 km) can be processed on an
ordinary workstation::

    acquire   -> tiles/<tx>_<ty>/stack.tif (cloud-free Sentinel-2 composite) + dem.tif
    train     -> model bundle from reference geological maps and/or drawn training areas
    classify  -> tiles/<tx>_<ty>/classified.tif + confidence.tif (seamless: reads neighbours' pixels)
    mosaic    -> mosaic/*.tif region-wide Cloud-Optimised GeoTIFFs with overviews (web map, GIS)
    stats     -> area of every lithology, per district when district boundaries are supplied
    export    -> GeoJSON polygons, PDF report
"""
from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.crs import CRS
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes, sieve
from rasterio.transform import Affine
from rasterio.warp import transform as warp_transform
from rasterio.warp import transform_geom
from rasterio.windows import Window

from .acquisition import (S2Catalog, Season, build_composite, fetch_dem, read_local_to_grid, utm_epsg,
                          write_composite)
from .config import ALGORITHM_LABELS, ALGORITHMS, CLASS_IDS, CLOUD_CLASS, class_name, hex_to_rgb, ALL_CLASSES
from .features import build_features
from .io import GeoInfo, align_to, class_colormap, write_raster
from .mapping import area_statistics_from_counts, hillshade_png
from .pipeline import (DEFAULT_PATCH_SIZE, ModelBundle, SampleCollector, Sources, fit_bundle, load_sources,
                       predict_window)

Progress = Callable[[float, str], None]
Log = Callable[[str], None]


def _noop(*_a, **_k):
    pass


# ---------------------------------------------------------------------------
# Configuration and grid
# ---------------------------------------------------------------------------

@dataclass
class RegionConfig:
    name: str
    aoi: dict                                   # GeoJSON geometry, EPSG:4326
    epsg: Optional[int] = None                  # default: UTM zone of the AOI centroid
    resolution: float = 20.0                    # metres
    tile_size: int = 1024                       # pixels (1024 x 20 m = 20.48 km)
    source: str = "sentinel2"                   # "sentinel2" (AWS, automatic) or "local"
    years: list = field(default_factory=lambda: [2023, 2024, 2025])
    months: list = field(default_factory=lambda: [7, 8, 9, 10])
    max_cloud: float = 30.0
    max_scenes: int = 6
    topo_correct: bool = True
    local_scenes: list = field(default_factory=list)   # for source="local": multi-band rasters
    local_sensor: str = "sentinel2"
    local_dem: Optional[str] = None
    stac_url: Optional[str] = None


@dataclass
class Tile:
    tx: int
    ty: int
    window: Window        # in region grid pixels
    info: GeoInfo

    @property
    def key(self) -> str:
        return f"{self.tx:03d}_{self.ty:03d}"


def _geom_bounds(geom: dict) -> tuple[float, float, float, float]:
    pts = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            pts.append(c)
        else:
            for cc in c:
                walk(cc)
    walk(geom["coordinates"])
    a = np.asarray(pts, dtype=float)
    return float(a[:, 0].min()), float(a[:, 1].min()), float(a[:, 0].max()), float(a[:, 1].max())


def load_geojson_geometry(data: dict) -> dict:
    """Accept a Geometry, Feature or FeatureCollection and return one (Multi)Polygon geometry."""
    if data.get("type") == "FeatureCollection":
        polys = []
        for f in data["features"]:
            g = f["geometry"]
            polys += [g["coordinates"]] if g["type"] == "Polygon" else list(g["coordinates"])
        return {"type": "MultiPolygon", "coordinates": polys} if len(polys) > 1 else {"type": "Polygon", "coordinates": polys[0]}
    if data.get("type") == "Feature":
        return data["geometry"]
    return data


class Region:
    STAGES = ("acquired", "classified")

    def __init__(self, folder: str | Path):
        self.folder = Path(folder)
        self.config = RegionConfig(**json.loads((self.folder / "region.json").read_text(encoding="utf-8")))
        g = json.loads((self.folder / "grid.json").read_text(encoding="utf-8"))
        self.crs = CRS.from_epsg(g["epsg"])
        self.transform = Affine(*g["transform"])
        self.width, self.height = g["width"], g["height"]
        self.tile_keys: list[str] = g["tiles"]
        self._lock = threading.Lock()

    # -- creation ---------------------------------------------------------------------
    @classmethod
    def create(cls, folder: str | Path, config: RegionConfig) -> "Region":
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        w, s, e, n = _geom_bounds(config.aoi)
        if config.epsg is None:
            config.epsg = utm_epsg((w + e) / 2, (s + n) / 2)
        crs = CRS.from_epsg(config.epsg)
        aoi = transform_geom("EPSG:4326", crs, config.aoi)
        x0, y0, x1, y1 = _geom_bounds(aoi)
        step = config.tile_size * config.resolution
        ox, oy = math.floor(x0 / step) * step, math.ceil(y1 / step) * step
        ncols, nrows = math.ceil((x1 - ox) / step), math.ceil((oy - y0) / step)
        coarse = Affine(step, 0, ox, 0, -step, oy)
        hit = rasterize([(aoi, 1)], out_shape=(nrows, ncols), transform=coarse, all_touched=True, dtype="uint8")
        tiles = [f"{tx:03d}_{ty:03d}" for ty, tx in zip(*np.nonzero(hit))]
        grid = {"epsg": config.epsg, "transform": list(Affine(config.resolution, 0, ox, 0, -config.resolution, oy))[:6],
                "width": ncols * config.tile_size, "height": nrows * config.tile_size, "tiles": tiles,
                "aoi_projected": aoi}
        (folder / "region.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")
        (folder / "grid.json").write_text(json.dumps(grid), encoding="utf-8")
        (folder / "state.json").write_text(json.dumps({"tiles": {}, "log": []}), encoding="utf-8")
        return cls(folder)

    # -- grid helpers -----------------------------------------------------------------
    @property
    def info(self) -> GeoInfo:
        return GeoInfo(self.transform, self.crs, self.width, self.height)

    def tile(self, key: str) -> Tile:
        tx, ty = (int(v) for v in key.split("_"))
        ts = self.config.tile_size
        win = Window(tx * ts, ty * ts, ts, ts)
        return Tile(tx, ty, win, self.info.window(win))

    def tiles(self) -> list[Tile]:
        return [self.tile(k) for k in self.tile_keys]

    def tile_dir(self, key: str) -> Path:
        return self.folder / "tiles" / key

    @property
    def aoi_projected(self) -> dict:
        return json.loads((self.folder / "grid.json").read_text(encoding="utf-8"))["aoi_projected"]

    def aoi_mask(self, info: GeoInfo) -> np.ndarray:
        return rasterize([(self.aoi_projected, 1)], out_shape=(info.height, info.width),
                         transform=info.transform, dtype="uint8").astype(bool)

    def area_km2(self) -> float:
        return float(len(self.tile_keys) * (self.config.tile_size * self.config.resolution / 1000) ** 2)

    def tile_geojson(self) -> dict:
        """Tile footprints (WGS84) with processing status, for the web map."""
        st = self.state()["tiles"]
        feats = []
        for t in self.tiles():
            left, top = t.info.transform @ (0, 0)
            right, bottom = t.info.transform @ (t.info.width, t.info.height)
            ring = [[left, bottom], [right, bottom], [right, top], [left, top], [left, bottom]]
            geom = transform_geom(self.crs, "EPSG:4326", {"type": "Polygon", "coordinates": [ring]})
            feats.append({"type": "Feature", "geometry": geom,
                          "properties": {"key": t.key, **st.get(t.key, {})}})
        return {"type": "FeatureCollection", "features": feats}

    # -- state ------------------------------------------------------------------------
    def state(self) -> dict:
        p = self.folder / "state.json"
        for i in range(40):   # the file may be mid-replace (Windows raises PermissionError)
            try:
                return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {"tiles": {}, "log": []}
            except (PermissionError, json.JSONDecodeError):
                time.sleep(0.05 * (i + 1))
        raise RuntimeError(f"cannot read {p}")

    def _update_tile(self, key: str, **values) -> None:
        with self._lock:
            st = self.state()
            st["tiles"].setdefault(key, {}).update(values)
            tmp = self.folder / f"state.json.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(st), encoding="utf-8")
            replace_file(tmp, self.folder / "state.json")

    def _set(self, **values) -> None:
        with self._lock:
            st = self.state()
            st.update(values)
            tmp = self.folder / f"state.json.{threading.get_ident()}.tmp"
            tmp.write_text(json.dumps(st), encoding="utf-8")
            replace_file(tmp, self.folder / "state.json")

    def summary(self) -> dict:
        st = self.state()["tiles"]
        n = len(self.tile_keys)
        return {"tiles": n,
                "acquired": sum(1 for k in self.tile_keys if st.get(k, {}).get("acquired")),
                "classified": sum(1 for k in self.tile_keys if st.get(k, {}).get("classified")),
                "failed": sum(1 for k in self.tile_keys if st.get(k, {}).get("error")),
                "area_km2": self.area_km2()}

    # -- 1. acquisition ---------------------------------------------------------------
    def acquire(self, progress: Progress = _noop, log: Log = print, keys: Optional[Sequence[str]] = None,
                force: bool = False) -> dict:
        cfg = self.config
        keys = list(keys or self.tile_keys)
        catalog = S2Catalog(cfg.stac_url, self.folder / "cache", log=log) if cfg.source == "sentinel2" else None
        season = Season(cfg.years, cfg.months, cfg.max_cloud)
        done = 0
        for i, key in enumerate(keys):
            progress(i / len(keys), f"Acquiring tile {key} ({i + 1}/{len(keys)})")
            st = self.state()["tiles"].get(key, {})
            if st.get("acquired") and not force:
                done += 1
                continue
            try:
                self._acquire_tile(self.tile(key), catalog, season, log)
                done += 1
            except Exception as e:  # noqa: BLE001 - record and continue with other tiles
                log(f"tile {key} failed: {type(e).__name__}: {e}")
                self._update_tile(key, error=f"acquire: {e}")
        self.build_vrts()
        progress(1.0, f"Acquired {done}/{len(keys)} tiles")
        return self.summary()

    def _acquire_tile(self, tile: Tile, catalog: Optional[S2Catalog], season: Season, log: Log) -> None:
        cfg = self.config
        d = self.tile_dir(tile.key)
        d.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        inside = self.aoi_mask(tile.info)
        if cfg.source == "local":
            dem = (align_to(cfg.local_dem, tile.info) if cfg.local_dem
                   else np.full((tile.info.height, tile.info.width), np.nan, np.float32))
            write_raster(d / "dem.tif", dem[None], tile.info, nodata=np.nan, descriptions=["elevation_m"])
            from .preprocessing import to_reflectance
            raw = read_local_to_grid(cfg.local_scenes, tile.info, 6)
            refl = to_reflectance(np.nan_to_num(raw), cfg.local_sensor) if cfg.local_sensor != "reflectance" else raw
            ok = np.all(np.isfinite(raw), 0) & inside
            stack = np.zeros((7, tile.info.height, tile.info.width), np.uint16)
            stack[:6] = np.where(ok[None], np.clip(np.nan_to_num(refl) * 10000, 1, 65535), 0)
            stack[6] = np.where(ok, 4, 0)
            from .acquisition import CompositeReport
            rep = CompositeReport(items=[str(p) for p in cfg.local_scenes], clear_fraction=float(ok.mean()),
                                  sun_azimuth=150.0, sun_elevation=55.0)
        else:
            dem = fetch_dem(tile.info, d / "dem.tif", log)
            items = catalog.search(tile.info.wgs84_bounds(), season)
            log(f"tile {tile.key}: {len(items)} candidate scenes")
            stack, rep = build_composite(tile.info, items, dem, cfg.max_scenes, topo_correct=cfg.topo_correct, log=log)
            stack[:, ~inside] = 0
        write_composite(d / "stack.tif", stack, tile.info, rep)
        meta = {"composite": rep.to_dict(), "seconds": round(time.time() - t0, 1)}
        (d / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self._update_tile(tile.key, acquired=True, error=None, clear=round(rep.clear_fraction, 3),
                          scenes=len(rep.items), dates=rep.dates, sun=[rep.sun_azimuth, rep.sun_elevation])

    def build_vrts(self) -> None:
        """Region-wide virtual mosaics of the tile stacks / DEMs (for seamless halo reads)."""
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("acquired")]
        if keys:
            _write_vrt(self.folder / "stack.vrt", self, keys, "stack.tif", 7, "UInt16", 0)
            _write_vrt(self.folder / "dem.vrt", self, keys, "dem.tif", 1, "Float32", "nan")

    # -- 2. training ------------------------------------------------------------------
    def training_labels(self, info: GeoInfo, vector_shapes: list, label_rasters: Sequence[Path] = ()) -> np.ndarray:
        lab = np.zeros((info.height, info.width), np.uint8)
        for r in label_rasters:
            part = np.nan_to_num(align_to(r, info, Resampling.nearest), nan=0).astype(np.uint8)
            lab = np.where(lab == 0, part, lab)
        left, top = info.transform @ (0, 0)
        right, bottom = info.transform @ (info.width, info.height)
        todo = [(g, v) for g, v, b in vector_shapes if b[0] < right and b[2] > left and b[1] < top and b[3] > bottom]
        if todo:
            vec = rasterize(todo, out_shape=lab.shape, transform=info.transform, fill=0, dtype="uint8")
            lab = np.where(vec > 0, vec, lab)   # drawn / vector labels override rasters
        return lab

    def prepare_shapes(self, features: Sequence[dict]) -> list:
        """[(geometry EPSG:4326, class_id)] -> [(geometry in region CRS, class_id, bounds)]."""
        out = []
        for geom, cid in features:
            if int(cid) not in CLASS_IDS:
                continue
            g = transform_geom("EPSG:4326", self.crs, geom)
            out.append((g, int(cid), _geom_bounds(g)))
        return out

    def train(self, out_dir, features: Sequence[tuple] = (), label_rasters: Sequence[Path] = (),
              algorithms: Sequence[str] = ALGORITHMS, samples_per_class: int = 4000, epochs: int = 30,
              seed: int = 0, name: Optional[str] = None, progress: Progress = _noop, log: Log = print) -> dict:
        """Train on every acquired tile that overlaps reference labels.

        ``features``: (GeoJSON geometry in WGS84, class id) pairs from a digitised geological map
        or from training areas drawn in the dashboard. ``label_rasters``: label GeoTIFFs (any CRS).
        """
        algorithms = [a for a in algorithms if a in ALGORITHMS]
        shapes_ = self.prepare_shapes(features)
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("acquired")]
        if not keys:
            raise ValueError("No acquired tiles - run data acquisition first")
        # find tiles that actually contain labels
        labelled = []
        for k in keys:
            lab = self.training_labels(self.tile(k).info, shapes_, label_rasters)
            if (lab > 0).sum() >= 20:
                labelled.append(k)
        if not labelled:
            raise ValueError("None of the acquired tiles overlaps the reference labels")
        log(f"{len(labelled)} tiles contain reference labels")
        col = SampleCollector(samples_per_class, patches="cnn" in algorithms, seed=seed)
        share = min(1.0, 3.0 / len(labelled))
        uses_dem = True
        for i, k in enumerate(labelled):
            progress(0.02 + 0.2 * i / len(labelled), f"Sampling tile {k} ({i + 1}/{len(labelled)})")
            t = self.tile(k)
            scene = load_sources(self._sources(k), t.window)
            lab = self.training_labels(t.info, shapes_, label_rasters)
            uses_dem = bool(uses_dem and scene.dem is not None and np.isfinite(scene.dem).any())
            feats, names = build_features(scene.reflectance, scene.dem if uses_dem else None, scene.pixel_size_m)
            n = col.add(feats, names, lab, scene.usable, share)
            log(f"tile {k}: {n} samples")
        meta = {"name": name or f"{self.config.name} model", "region": str(self.folder), "sensor": "sentinel2_composite",
                "dos": False, "uses_dem": uses_dem, "masking": True, "topo_correct": False,
                "tiles_used": labelled}
        meta = fit_bundle(col, out_dir, algorithms, meta, epochs, seed, progress, 0.25, 0.73)
        progress(1.0, "Training complete")
        return meta

    # -- 3. classification ------------------------------------------------------------
    def _sources(self, key: str) -> Sources:
        sun = self.state()["tiles"].get(key, {}).get("sun")
        return Sources(self.folder / "stack.vrt", self.folder / "dem.vrt", sensor="sentinel2_composite",
                       sun=tuple(sun) if sun else None, masking=True)

    def classify(self, model_dir, algo: Optional[str] = None, smoothing: int = 3, progress: Progress = _noop,
                 log: Log = print, keys: Optional[Sequence[str]] = None, force: bool = True) -> dict:
        bundle = ModelBundle(model_dir)
        algo = algo or bundle.best_algorithm()
        st = self.state()["tiles"]
        keys = [k for k in (keys or self.tile_keys) if st.get(k, {}).get("acquired")]
        if not keys:
            raise ValueError("No acquired tiles to classify")
        self.build_vrts()
        halo = DEFAULT_PATCH_SIZE // 2 + max(1, smoothing) + 2
        for i, k in enumerate(keys):
            if not force and st.get(k, {}).get("classified"):
                continue
            progress(i / len(keys), f"Classifying tile {k} ({i + 1}/{len(keys)}) with {ALGORITHM_LABELS[algo]}")
            t = self.tile(k)
            try:
                labels, conf, _scene, _ = predict_window(bundle, algo, self._sources(k), t.window, halo,
                                                         (self.height, self.width), smoothing)
                inside = self.aoi_mask(t.info)
                labels[~inside] = 0
                conf[~inside] = 0
                d = self.tile_dir(k)
                write_raster(d / "classified.tif", labels[None], t.info, nodata=0, colormap=class_colormap())
                write_raster(d / "confidence.tif", np.round(conf * 100).astype(np.uint8)[None], t.info)
                counts = np.bincount(labels.ravel(), minlength=256)
                np.save(d / "counts.npy", counts)
                self._update_tile(k, classified=True, error=None, algorithm=algo)
            except Exception as e:  # noqa: BLE001
                log(f"tile {k} failed: {type(e).__name__}: {e}")
                self._update_tile(k, error=f"classify: {e}")
        self._set(model=str(model_dir), algorithm=algo, classified_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        progress(1.0, "Classification complete")
        return self.summary()

    # -- 4. mosaics -------------------------------------------------------------------
    def build_mosaics(self, progress: Progress = _noop) -> dict[str, Path]:
        """Region-wide tiled GeoTIFFs with internal overviews (used by the web map and for GIS)."""
        out = self.folder / "mosaic"
        out.mkdir(exist_ok=True)
        st = self.state()["tiles"]
        acquired = [k for k in self.tile_keys if st.get(k, {}).get("acquired")]
        classified = [k for k in self.tile_keys if st.get(k, {}).get("classified")]
        analysed = [k for k in self.tile_keys if (self.tile_dir(k) / "analytics.tif").exists()]
        products = {"rgb": (3, acquired), "falsecolor": (3, acquired), "hillshade": (1, acquired),
                    "surface": (1, acquired), "lithology": (1, classified), "confidence": (1, classified),
                    "alteration": (1, analysed), "hazard": (1, analysed), "clusters": (1, analysed)}
        gem_tiles = [k for k in self.tile_keys if (self.tile_dir(k) / "gems.tif").exists()]
        from .gems import GEM_MODELS
        products["gems"] = (1, gem_tiles)
        for m in GEM_MODELS:
            products[f"gem_{m.key}"] = (1, gem_tiles)
        if gem_tiles and all(_band_count(self.tile_dir(k) / "gems.tif") >= 7 for k in gem_tiles):
            products["gem_ml"] = (1, gem_tiles)
        paths = {}
        for pi, (name, (count, keys)) in enumerate(products.items()):
            if not keys:
                continue
            path = out / f"{name}.tif"
            final = path
            path = out / f"{name}.building.tif"   # build aside, then swap in (open handles on Windows)
            prof = dict(driver="GTiff", width=self.width, height=self.height, count=count, dtype="uint8",
                        crs=self.crs, transform=self.transform, nodata=0, compress="deflate", tiled=True,
                        blockxsize=512, blockysize=512, BIGTIFF="IF_SAFER")
            if name in ("rgb", "falsecolor"):
                prof.update(photometric="RGB")
            with rasterio.open(path, "w", **prof) as dst:
                for i, k in enumerate(keys):
                    progress((pi + i / len(keys)) / len(products), f"Mosaicking {name}: tile {k}")
                    dst.write(self._render_tile(k, name), window=self.tile(k).window)
                if name in ("lithology", "surface"):
                    dst.write_colormap(1, class_colormap())
                factors = [f for f in (2, 4, 8, 16, 32, 64) if max(self.width, self.height) / f >= 256]
                if factors:
                    rs = (Resampling.nearest if name in ("lithology", "surface", "hazard", "clusters")
                          else Resampling.average)
                    dst.build_overviews(factors, rs)
                    dst.update_tags(ns="rio_overview", resampling=rs.name)
            from .tiles import release
            release(final)
            replace_file(path, final)
            paths[name] = final
        self._set(mosaic_built=time.strftime("%Y-%m-%d %H:%M:%S"), mosaic_version=int(time.time()))
        progress(1.0, "Mosaics built")
        return paths

    def _render_tile(self, key: str, name: str) -> np.ndarray:
        d = self.tile_dir(key)
        if name.startswith("gem"):
            from .gems import GEM_MODELS
            band = 1 if name == "gems" else 7 if name == "gem_ml" else \
                3 + [m.key for m in GEM_MODELS].index(name[4:])
            with rasterio.open(d / "gems.tif") as src:
                return src.read(band)[None]
        if name in ("alteration", "hazard", "clusters"):
            band = {"alteration": 1, "hazard": 3, "clusters": 4}[name]
            with rasterio.open(d / "analytics.tif") as src:
                return src.read(band)[None]
        if name == "surface":
            # snow / water / vegetation / shadow straight from the imagery (no model needed); 1 = bare ground
            t = self.tile(key)
            scene = load_sources(self._sources(key), t.window)
            out = np.where(scene.landcover > 0, scene.landcover, 1).astype(np.uint8)
            out[~scene.valid] = CLOUD_CLASS
            out[~self.aoi_mask(t.info)] = 0
            return out[None]
        if name in ("lithology", "confidence"):
            with rasterio.open(d / ("classified.tif" if name == "lithology" else "confidence.tif")) as src:
                a = src.read(1)
            if name == "confidence":
                with rasterio.open(d / "classified.tif") as src:
                    lab = src.read(1)
                a = np.where(np.isin(lab, CLASS_IDS), np.maximum(a, 1), 0).astype(np.uint8)
            return a[None]
        if name == "hillshade":
            with rasterio.open(d / "dem.tif") as src:
                dem = src.read(1)
            ok = np.isfinite(dem)
            if not ok.any():
                return np.zeros((1, *dem.shape), np.uint8)
            hs = hillshade_png(np.where(ok, dem, np.nanmean(dem)), self.config.resolution)[..., 0]
            return np.where(ok, np.maximum(hs, 1), 0).astype(np.uint8)[None]
        with rasterio.open(d / "stack.tif") as src:
            bands = (3, 2, 1) if name == "rgb" else (6, 4, 1)
            a = src.read(list(bands)).astype(np.float32) / 10000.0
        ok = np.all(a > 0, axis=0)
        # fixed, region-wide stretch so tiles match (no per-tile percentile seams)
        hi = 0.32 if name == "rgb" else 0.45
        v = np.clip(a / hi, 0, 1) ** (1 / 1.4)
        return np.where(ok[None], np.clip(v * 254 + 1, 1, 255), 0).astype(np.uint8)

    # -- 5. statistics ----------------------------------------------------------------
    def statistics(self, districts: Optional[dict] = None, name_field: str = "name") -> dict:
        """Region totals and (optionally) per-district areas of every class."""
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("classified")]
        px_km2 = (self.config.resolution / 1000) ** 2
        total = np.zeros(256, np.int64)
        for k in keys:
            p = self.tile_dir(k) / "counts.npy"
            if p.exists():
                total += np.load(p)
        result = {"region": area_statistics_from_counts(total, px_km2), "tiles": len(keys),
                  "classified_km2": float(total[1:].sum() * px_km2), "districts": []}
        if districts:
            feats = districts.get("features", [])
            dshapes = [(transform_geom("EPSG:4326", self.crs, f["geometry"]), i + 1) for i, f in enumerate(feats)]
            per = np.zeros((len(feats) + 1, 256), np.int64)
            for k in keys:
                t = self.tile(k)
                dmap = rasterize(dshapes, out_shape=(t.info.height, t.info.width), transform=t.info.transform,
                                 fill=0, dtype="uint16")
                with rasterio.open(self.tile_dir(k) / "classified.tif") as src:
                    lab = src.read(1)
                per += np.bincount(dmap.astype(np.int64).ravel() * 256 + lab.ravel(),
                                   minlength=per.size).reshape(per.shape)
            for i, f in enumerate(feats):
                props = f.get("properties") or {}
                dname = props.get(name_field) or props.get("NAME") or props.get("name") or f"District {i + 1}"
                result["districts"].append({"name": str(dname),
                                            "stats": area_statistics_from_counts(per[i + 1], px_km2),
                                            "km2": float(per[i + 1, 1:].sum() * px_km2)})
        return result

    # -- 6. point query ---------------------------------------------------------------
    def query(self, lon: float, lat: float) -> dict:
        xs, ys = warp_transform("EPSG:4326", self.crs, [lon], [lat])
        col, row = ~self.transform @ (xs[0], ys[0])
        col, row = int(math.floor(col)), int(math.floor(row))
        ts = self.config.tile_size
        key = f"{col // ts:03d}_{row // ts:03d}"
        out = {"lon": lon, "lat": lat, "tile": key, "inside": key in self.tile_keys}
        if not out["inside"] or col < 0 or row < 0:
            return out
        d = self.tile_dir(key)
        r, c = row % ts, col % ts
        w = Window(c, r, 1, 1)

        def px(name, band=1):
            p = d / name
            if not p.exists():
                return None
            with rasterio.open(p) as src:
                return src.read(band, window=w)[0, 0].item()
        cls = px("classified.tif")
        if cls is not None:
            out.update(class_id=int(cls), class_name=class_name(cls) if cls else "Outside area",
                       color=ALL_CLASSES[cls].color if cls in ALL_CLASSES else None, confidence=px("confidence.tif"))
        elev = px("dem.tif")
        if elev is not None and np.isfinite(elev):
            out["elevation_m"] = round(float(elev), 1)
        if (d / "stack.tif").exists():
            with rasterio.open(d / "stack.tif") as src:
                v = src.read(list(range(1, 7)), window=w)[:, 0, 0] / 10000.0
            out["reflectance"] = dict(zip(["blue", "green", "red", "nir", "swir1", "swir2"], np.round(v, 4).tolist()))
        return out

    # -- 7. vector export -------------------------------------------------------------
    def export_geojson(self, out_path, min_pixels: int = 25, include_masks: bool = False,
                       progress: Progress = _noop) -> Path:
        """Polygonise the classified tiles to one GeoJSON (WGS84), streaming to disk.

        Patches smaller than ``min_pixels`` (default 25 px = 1 ha at 20 m) are removed with a
        sieve filter first. Polygons are split at tile edges (``tile`` property).
        """
        out_path = Path(out_path)
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("classified")]
        px_ha = (self.config.resolution ** 2) / 1e4
        n = 0
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write('{"type": "FeatureCollection", "features": [\n')
            for i, k in enumerate(keys):
                progress(i / max(1, len(keys)), f"Polygonising tile {k}")
                t = self.tile(k)
                with rasterio.open(self.tile_dir(k) / "classified.tif") as src:
                    lab = src.read(1)
                keep = np.isin(lab, CLASS_IDS if not include_masks else list(ALL_CLASSES))
                lab = np.where(keep, lab, 0).astype(np.uint8)
                if min_pixels > 1:
                    lab = sieve(lab, size=min_pixels, mask=lab > 0)
                for geom, val in shapes(lab, mask=lab > 0, transform=t.info.transform):
                    g = transform_geom(self.crs, "EPSG:4326", geom, precision=6)
                    area = _ring_area(geom) / 1e4
                    feat = {"type": "Feature", "geometry": g,
                            "properties": {"class_id": int(val), "lithology": class_name(int(val)),
                                           "area_ha": round(area, 2), "tile": k}}
                    fh.write((",\n" if n else "") + json.dumps(feat))
                    n += 1
            fh.write("\n]}\n")
        progress(1.0, f"Exported {n} polygons")
        return out_path

    # -- 8. analytics (no training data needed) ---------------------------------------
    def _read_tile(self, key: str, halo: int = 0):
        """Scene data for a tile plus ``halo`` pixels of its neighbours; returns (tile, scene, core slice)."""
        t = self.tile(key)
        ts = self.config.tile_size
        x0, y0 = max(0, int(t.window.col_off) - halo), max(0, int(t.window.row_off) - halo)
        x1 = min(self.width, int(t.window.col_off) + ts + halo)
        y1 = min(self.height, int(t.window.row_off) + ts + halo)
        scene = load_sources(self._sources(key), Window(x0, y0, x1 - x0, y1 - y0))
        oy, ox = int(t.window.row_off) - y0, int(t.window.col_off) - x0
        return t, scene, (slice(oy, oy + ts), slice(ox, ox + ts))

    def analytics(self) -> Optional[dict]:
        p = self.folder / "products" / "analytics.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def analyze(self, n_clusters: int = 10, progress: Progress = _noop, log: Log = print,
                samples_per_tile: int = 4000, seed: int = 0) -> dict:
        """Mineral-alteration anomalies + targets, landslide susceptibility and spectral units.

        Writes ``tiles/*/analytics.tif`` (band 1 alteration score+1, 2 dominant alteration type,
        3 susceptibility class, 4 spectral unit, 5 unit confidence %, 6 surface code) and
        ``products/analytics.json``, ``targets.geojson``, ``targets.csv``.
        """
        import joblib
        from .analytics import (ALTERATION_INDICES, CLUSTER_PALETTE, HAZARD_CLASSES, RobustStats,
                                alteration_indices, alteration_score, find_targets, fit_clusters,
                                landslide_susceptibility, predict_clusters, spectral_features)
        from .config import SNOW_CLASS, WATER_CLASS
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("acquired")]
        if not keys:
            raise ValueError("No acquired tiles - run data acquisition first")
        self.build_vrts()
        rng = np.random.default_rng(seed)
        # pass 1: region-wide statistics and cluster model from a pixel sample
        idx_s = {k: [] for k in ALTERATION_INDICES}
        feat_s = []
        for i, key in enumerate(keys):
            progress(0.02 + 0.25 * i / len(keys), f"Sampling spectra: tile {key} ({i + 1}/{len(keys)})")
            _t, scene, _sl = self._read_tile(key)
            r, c = np.nonzero(scene.usable)
            if len(r) < 50:
                continue
            take = rng.choice(len(r), min(samples_per_tile, len(r)), replace=False)
            r, c = r[take], c[take]
            ind = alteration_indices(scene.reflectance)
            for k in ALTERATION_INDICES:
                idx_s[k].append(ind[k][r, c])
            feat_s.append(spectral_features(scene.reflectance)[:, r, c].T)
        if not feat_s:
            raise ValueError("No usable (snow-, cloud- and vegetation-free) pixels found")
        stats = RobustStats.fit({k: np.concatenate(v) for k, v in idx_s.items()})
        samples = np.concatenate(feat_s)
        k = int(max(2, min(n_clusters, 16, len(samples) // 50)))
        progress(0.3, f"Clustering {len(samples):,} pixels into {k} spectral units")
        mean, std, km = fit_clusters(samples, k, seed)
        adir = self.folder / "analytics"
        adir.mkdir(exist_ok=True)
        joblib.dump({"mean": mean, "std": std, "km": km, "stats": stats.to_dict()}, adir / "model.joblib")

        # pass 2: per-tile products
        px_km2 = (self.config.resolution / 1000) ** 2
        targets: list[dict] = []
        hz_counts = np.zeros(6, np.int64)
        cl_n = np.zeros(k + 1, np.int64)
        cl_refl = np.zeros((k + 1, 6))
        cl_elev = np.zeros(k + 1)
        for i, key in enumerate(keys):
            progress(0.32 + 0.6 * i / len(keys), f"Analysing tile {key} ({i + 1}/{len(keys)})")
            t, scene, sl = self._read_tile(key, halo=16)
            usable = scene.usable
            lc = scene.landcover if scene.landcover is not None else np.zeros(usable.shape, np.uint8)
            # keep alteration away from snow / cloud / no-data edges (mixed pixels give false anomalies)
            from scipy import ndimage as ndi
            edge = ndi.binary_dilation((lc == SNOW_CLASS) | ~scene.valid, iterations=5)
            score, dom = alteration_score(scene.reflectance, usable & ~edge, stats)
            lith_h = np.zeros(usable.shape, np.uint8)
            cpath = self.tile_dir(key) / "classified.tif"
            if cpath.exists():
                with rasterio.open(cpath) as src:
                    lith_h[sl] = src.read(1)
            has_dem = scene.dem is not None and np.isfinite(scene.dem).any()
            if has_dem:
                refl = np.nan_to_num(scene.reflectance)
                ndvi = (refl[3] - refl[2]) / (refl[3] + refl[2] + 1e-6)
                _idx, hz = landslide_susceptibility(scene.dem, self.config.resolution, lc == WATER_CLASS, ndvi,
                                                    lith_h)
                hz[~scene.valid | np.isin(lc, [SNOW_CLASS, WATER_CLASS])] = 0
            else:
                hz = np.zeros(usable.shape, np.uint8)
            clusters, cconf = predict_clusters(spectral_features(scene.reflectance), usable, mean, std, km)
            surface = np.where(scene.valid, lc, CLOUD_CLASS).astype(np.uint8)
            inside = self.aoi_mask(t.info)
            core = [a[sl] for a in (score, dom, hz, clusters, cconf, surface)]
            score_c, dom_c, hz_c, cl_c, conf_c, surf_c = core
            for a in core:
                a[~inside] = 0
            band1 = np.where(score_c > 0, np.round(score_c) + 1, 0).astype(np.uint8)
            stack = np.stack([band1, dom_c, hz_c, cl_c, np.round(conf_c * 100).astype(np.uint8), surf_c])
            write_raster(self.tile_dir(key) / "analytics.tif", stack.astype(np.uint8), t.info, nodata=0,
                         descriptions=["alteration_score_plus1", "alteration_type", "landslide_class",
                                       "spectral_unit", "unit_confidence", "surface_code"])
            dem_c = scene.dem[sl] if has_dem else np.full(score_c.shape, np.nan, np.float32)
            targets += [dict(tt, tile=key) for tt in
                        find_targets(score_c, dom_c, t.info.transform, pixel_area_ha=px_km2 * 100,
                                     extra={"elevation_m": dem_c, "lithology_id": lith_h[sl]})]
            hz_counts += np.bincount(hz_c.ravel(), minlength=6)[:6]
            cl_n += np.bincount(cl_c.ravel(), minlength=k + 1)[:k + 1]
            refl_c = np.nan_to_num(scene.reflectance[:, sl[0], sl[1]])
            for b in range(6):
                cl_refl[:, b] += np.bincount(cl_c.ravel(), weights=refl_c[b].ravel(), minlength=k + 1)[:k + 1]
            cl_elev += np.bincount(cl_c.ravel(), weights=np.nan_to_num(dem_c).ravel(), minlength=k + 1)[:k + 1]
            self._update_tile(key, analysed=True)

        progress(0.94, "Ranking exploration targets")
        targets.sort(key=lambda t: -t["rank_score"])
        targets = targets[:1000]
        if targets:
            lons, lats = warp_transform(self.crs, "EPSG:4326", [t["x"] for t in targets], [t["y"] for t in targets])
            for n, (t, lo, la) in enumerate(zip(targets, lons, lats), start=1):
                t.update(id=n, lon=round(lo, 6), lat=round(la, 6))
                if t.get("lithology_id") in CLASS_IDS:
                    t["lithology"] = class_name(t["lithology_id"])
        out = self.folder / "products"
        out.mkdir(exist_ok=True)
        _write_targets(out, targets)
        clusters_info = []
        for c in range(1, k + 1):
            n = int(cl_n[c])
            clusters_info.append({"id": c, "color": CLUSTER_PALETTE[(c - 1) % len(CLUSTER_PALETTE)], "pixels": n,
                                  "area_km2": round(n * px_km2, 2),
                                  "spectrum": [round(v / n, 4) if n else 0 for v in cl_refl[c]],
                                  "mean_elevation_m": round(cl_elev[c] / n, 0) if n else None, "class_id": None})
        result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "n_clusters": k, "stats": stats.to_dict(),
                  "clusters": clusters_info,
                  "hazard_km2": {str(c): round(float(hz_counts[c] * px_km2), 2) for c in HAZARD_CLASSES},
                  "targets_total": len(targets), "top_target": targets[0] if targets else None,
                  "targets_top": targets[:25]}
        (out / "analytics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._set(analysed_at=result["created"])
        progress(1.0, f"Analytics complete: {len(targets)} alteration targets, {k} spectral units")
        return result

    def label_clusters(self, mapping: dict, progress: Progress = _noop, log: Log = print) -> dict:
        """Turn spectral units into a lithology map: ``mapping`` = {unit id: rock class id}."""
        mapping = {int(a): int(b) for a, b in mapping.items() if b and int(b) in CLASS_IDS}
        if not mapping:
            raise ValueError("Assign at least one spectral unit to a rock class")
        info = self.analytics() or {}
        lut = np.zeros(256, np.uint8)
        for a, b in mapping.items():
            lut[a] = b
        keys = [k for k in self.tile_keys if (self.tile_dir(k) / "analytics.tif").exists()]
        for i, key in enumerate(keys):
            progress(i / max(1, len(keys)), f"Labelling tile {key}")
            t = self.tile(key)
            with rasterio.open(self.tile_dir(key) / "analytics.tif") as src:
                cl, conf, surf = src.read(4), src.read(5), src.read(6)
            lab = lut[cl]
            masked = (cl == 0) & (surf > 0)
            lab[masked] = surf[masked]
            conf = np.where(np.isin(lab, CLASS_IDS), conf, 0).astype(np.uint8)
            d = self.tile_dir(key)
            write_raster(d / "classified.tif", lab[None], t.info, nodata=0, colormap=class_colormap())
            write_raster(d / "confidence.tif", conf[None], t.info)
            np.save(d / "counts.npy", np.bincount(lab.ravel(), minlength=256))
            self._update_tile(key, classified=True, error=None, algorithm="units")
        for c in info.get("clusters", []):
            c["class_id"] = mapping.get(c["id"])
        if info:
            (self.folder / "products" / "analytics.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
        self._set(model=None, algorithm="units", classified_at=time.strftime("%Y-%m-%d %H:%M:%S"))
        progress(1.0, "Lithology map created from labelled spectral units")
        return self.summary()

    def field_validation(self, observations: Sequence[dict]) -> dict:
        """Compare field observations (lat, lon, class_id) with the current lithology map."""
        from .evaluation import evaluate
        rows, truth, pred = [], [], []
        for o in observations:
            q = self.query(float(o["lon"]), float(o["lat"]))
            p = q.get("class_id")
            ok = p in CLASS_IDS
            rows.append({"id": o.get("id"), "observed": int(o["class_id"]), "predicted": p if ok else None,
                         "match": bool(ok and p == int(o["class_id"]))})
            if ok:
                truth.append(int(o["class_id"]))
                pred.append(p)
        m = evaluate(np.asarray(truth), np.asarray(pred)) if truth else None
        return {"points": rows, "n_compared": len(truth), "metrics": m}



    # -- 9. gemstone prospectivity -----------------------------------------------------
    def gems(self) -> Optional[dict]:
        p = self.folder / "products" / "gems.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None

    def gem_occurrences(self) -> list[dict]:
        p = self.folder / "gem_occurrences.json"
        return json.loads(p.read_text(encoding="utf-8")) if p.exists() else []

    def set_gem_occurrences(self, occurrences: list[dict]) -> None:
        (self.folder / "gem_occurrences.json").write_text(json.dumps(occurrences, indent=1), encoding="utf-8")

    def _cos_i(self, key: str, scene):
        from .preprocessing import illumination
        sun = self.state()["tiles"].get(key, {}).get("sun")
        if scene.dem is None or not sun or not np.isfinite(scene.dem).any():
            return None
        return illumination(scene.dem, self.config.resolution, sun[0], sun[1])

    def _gem_usable(self, scene):
        """Bare, snow-free, sparsely vegetated rock with a buffer around snow and data gaps."""
        from scipy import ndimage as ndi
        from .config import SNOW_CLASS
        r = np.nan_to_num(scene.reflectance)
        ndvi = (r[3] - r[2]) / (r[3] + r[2] + 1e-6)
        lc = scene.landcover if scene.landcover is not None else np.zeros(ndvi.shape, np.uint8)
        edge = ndi.binary_dilation((lc == SNOW_CLASS) | ~scene.valid, iterations=5)
        return scene.usable & ~edge & (ndvi <= 0.3) & (r.mean(axis=0) >= 0.06)

    def gem_analysis(self, occurrences: Optional[list[dict]] = None, progress: Progress = _noop,
                     log: Log = print, samples_per_tile: int = 4000, seed: int = 0) -> dict:
        """Gemstone prospectivity for every acquired tile (see :mod:`rockmap.gems`).

        Writes ``tiles/*/gems.tif`` (band 1 best score+1, 2 best model, 3-6 model scores+1,
        7 data-driven score+1 when trained) and ``products/gems.json``, ``gem_targets.csv/.geojson``.
        """
        from .gems import (FEATURE_KEYS, GEM_MODELS, EVIDENCE, EvidenceStats, bedrock_mask, evidence_layers,
                           find_gem_targets, model_scores, raw_evidence, success_rates)
        occurrences = list(occurrences if occurrences is not None else self.gem_occurrences())
        st = self.state()["tiles"]
        keys = [k for k in self.tile_keys if st.get(k, {}).get("acquired")]
        if not keys:
            raise ValueError("No acquired tiles - run data acquisition first")
        self.build_vrts()
        rng = np.random.default_rng(seed)
        ts = self.config.tile_size
        # occurrences -> region pixel coordinates
        occ_px = []
        if occurrences:
            xs, ys = warp_transform("EPSG:4326", self.crs, [o["lon"] for o in occurrences],
                                    [o["lat"] for o in occurrences])
            for o, x, y in zip(occurrences, xs, ys):
                c, r = ~self.transform @ (x, y)
                occ_px.append((int(r), int(c)))

        # pass 1: robust region-wide statistics of the spectral evidence
        samples = {k: [] for k in EVIDENCE}
        for i, key in enumerate(keys):
            progress(0.02 + 0.2 * i / len(keys), f"Gem evidence statistics: tile {key} ({i + 1}/{len(keys)})")
            _t, scene, _sl = self._read_tile(key)
            usable = self._gem_usable(scene)
            # statistics describe *bedrock*: fields, orchards, fans and terraces would widen the
            # spread of the band ratios and hide the weak anomalies of marble and pegmatite
            r6 = np.nan_to_num(scene.reflectance)
            ndvi = (r6[3] - r6[2]) / (r6[3] + r6[2] + 1e-6)
            rock = bedrock_mask(scene.dem, self.config.resolution)
            fit_mask = usable & (ndvi < 0.2) & (rock if rock is not None else True)
            rr, cc = np.nonzero(fit_mask if fit_mask.sum() >= 500 else usable)
            if len(rr) < 50:
                continue
            take = rng.choice(len(rr), min(samples_per_tile, len(rr)), replace=False)
            raw = raw_evidence(scene.reflectance, self._cos_i(key, scene), usable)
            for k in EVIDENCE:
                samples[k].append(raw[k][rr[take], cc[take]])
        if not samples["brightness"]:
            raise ValueError("No usable (snow-, cloud- and vegetation-free) rock pixels")
        stats = EvidenceStats.fit({k: np.concatenate(v) for k, v in samples.items()})

        # pass 2: evidence, model scores, targets
        n_models = len(GEM_MODELS)
        px_km2 = (self.config.resolution / 1000) ** 2
        high = np.zeros(n_models + 1)
        background = [[] for _ in range(n_models + 1)]
        occ_scores = [[None] * (n_models + 1) for _ in occurrences]
        occ_feats = [None] * len(occurrences)
        bg_feats = []
        targets: list[dict] = []
        for i, key in enumerate(keys):
            progress(0.25 + 0.55 * i / len(keys), f"Gem prospectivity: tile {key} ({i + 1}/{len(keys)})")
            t, scene, sl = self._read_tile(key, halo=16)
            usable = self._gem_usable(scene)
            lith = None
            cpath = self.tile_dir(key) / "classified.tif"
            if cpath.exists():
                lith = np.zeros(usable.shape, np.uint8)
                with rasterio.open(cpath) as src:
                    lith[sl] = src.read(1)
            ev = evidence_layers(scene.reflectance, usable, stats, self.config.resolution, lith, scene.dem,
                                 self._cos_i(key, scene))
            scores = model_scores(ev)
            core_ok = usable[sl] & self.aoi_mask(t.info)
            sc = np.where(core_ok[None], scores[:, sl[0], sl[1]], 0)
            best = sc.max(axis=0)
            dom = np.where(core_ok, np.argmax(sc, axis=0) + 1, 0).astype(np.uint8)
            allsc = np.concatenate([best[None], sc])
            bands = [np.where(core_ok, np.round(best) + 1, 0), dom] + \
                    [np.where(core_ok, np.round(sc[m]) + 1, 0) for m in range(n_models)]
            write_raster(self.tile_dir(key) / "gems.tif", np.stack(bands).astype(np.uint8), t.info, nodata=0,
                         descriptions=["gem_score_plus1", "gem_model"] + [f"{m.key}_plus1" for m in GEM_MODELS])
            for m in range(n_models + 1):
                high[m] += float(((allsc[m] >= 75) & core_ok).sum()) * px_km2
            rr, cc = np.nonzero(core_ok)
            if len(rr):
                take = rng.choice(len(rr), min(3000, len(rr)), replace=False)
                for m in range(n_models + 1):
                    background[m].append(allsc[m][rr[take], cc[take]])
                bg_feats.append(np.stack([ev[f][sl][rr[take], cc[take]] for f in FEATURE_KEYS], 1))
            y0, x0 = int(t.window.row_off), int(t.window.col_off)
            for j, (r, c) in enumerate(occ_px):
                if y0 <= r < y0 + ts and x0 <= c < x0 + ts:
                    lr, lc = r - y0, c - x0
                    r0, r1, c0, c1 = max(0, lr - 1), lr + 2, max(0, lc - 1), lc + 2   # 3x3 window: GPS / map error
                    occ_scores[j] = [float(allsc[m][r0:r1, c0:c1].max()) for m in range(n_models + 1)]
                    win = [ev[f][sl][r0:r1, c0:c1].reshape(-1) for f in FEATURE_KEYS]
                    occ_feats[j] = np.stack(win, 1)
            self._update_tile(key, gems=True)

        bg = [np.concatenate(b) if b else np.zeros(0) for b in background]
        # validation with known occurrences
        validation = {"overall": success_rates([s[0] for s in occ_scores if s[0] is not None], bg[0])}
        for mi, m in enumerate(GEM_MODELS, start=1):
            vals = [s[mi] for o, s in zip(occurrences, occ_scores) if s[0] is not None and o.get("model") == m.key]
            validation[m.key] = success_rates(vals, bg[mi])
        inside = sum(1 for s in occ_scores if s[0] is not None)

        # optional data-driven model (presence / background Random Forest)
        ml = self._gem_ml(occ_feats, bg_feats, keys, stats, progress, log, seed) if inside >= 8 else None

        from .gems import MAX_TARGETS_PER_MODEL, target_cutoffs
        cutoffs = target_cutoffs({m.key: bg[i + 1] for i, m in enumerate(GEM_MODELS)})
        # targets = compact zones above each model's own top-1 % threshold
        progress(0.96, "Delineating gem targets")
        for key in keys:
            t = self.tile(key)
            with rasterio.open(self.tile_dir(key) / "gems.tif") as src:
                sc = np.clip(src.read(list(range(3, 3 + n_models))).astype(np.float32) - 1, 0, None)
            dem_p = self.tile_dir(key) / "dem.tif"
            dem_c = None
            if dem_p.exists():
                with rasterio.open(dem_p) as src:
                    dem_c = src.read(1)
            for mi, m in enumerate(GEM_MODELS):
                only = np.zeros_like(sc)
                only[mi] = sc[mi]
                found = find_gem_targets(only, t.info.transform, px_km2 * 100, threshold=cutoffs[m.key],
                                         elevation=dem_c)
                targets += [dict(tt, tile=key) for tt in found if tt["model"] == m.key]
        ranked = []
        for m in GEM_MODELS:   # best first, at least 500 m apart, so one outcrop is not listed ten times
            kept = []
            for t in sorted((t for t in targets if t["model"] == m.key), key=lambda t: -t["rank_score"]):
                if all(math.hypot(t["x"] - k["x"], t["y"] - k["y"]) >= 500 for k in kept):
                    kept.append(t)
                if len(kept) >= MAX_TARGETS_PER_MODEL:
                    break
            ranked += kept
        targets = sorted(ranked, key=lambda t: -t["rank_score"])
        if targets:
            lons, lats = warp_transform(self.crs, "EPSG:4326", [t["x"] for t in targets], [t["y"] for t in targets])
            for n, (t, lo, la) in enumerate(zip(targets, lons, lats), start=1):
                t.update(id=n, lon=round(lo, 6), lat=round(la, 6))
        out = self.folder / "products"
        out.mkdir(exist_ok=True)
        _write_gem_targets(out, targets)
        per_model = []
        for mi, m in enumerate(GEM_MODELS, start=1):
            per_model.append({"key": m.key, "name": m.name, "gems": m.gems, "color": m.color, "setting": m.setting,
                              "high_km2": round(high[mi], 2), "target_cutoff": round(cutoffs[m.key], 1),
                              "targets": sum(1 for t in targets if t["model"] == m.key),
                              "validation": validation[m.key]})
        result = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "stats": stats.to_dict(), "models": per_model,
                  "high_km2": round(high[0], 2), "targets_total": len(targets), "targets_top": targets[:30],
                  "occurrences": len(occurrences), "occurrences_inside": inside, "validation": validation,
                  "ml": ml}
        (out / "gems.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        self._set(gems_at=result["created"])
        progress(1.0, f"Gem prospectivity complete: {len(targets)} targets")
        return result

    def _gem_ml(self, occ_feats, bg_feats, keys, stats, progress, log, seed) -> Optional[dict]:
        """Random Forest trained on known occurrences vs. random background, with grouped CV."""
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.metrics import roc_auc_score
        from sklearn.model_selection import GroupKFold
        from .gems import FEATURE_KEYS, evidence_layers
        pos = [(j, f) for j, f in enumerate(occ_feats) if f is not None]
        X_pos = np.concatenate([f for _, f in pos])
        g_pos = np.concatenate([np.full(len(f), j) for j, f in pos])
        bgf = np.concatenate(bg_feats)
        rng = np.random.default_rng(seed)
        bgf = bgf[rng.choice(len(bgf), min(len(bgf), 20 * len(X_pos)), replace=False)]
        X = np.concatenate([X_pos, bgf])
        y = np.concatenate([np.ones(len(X_pos)), np.zeros(len(bgf))])
        groups = np.concatenate([g_pos, 10_000 + rng.integers(0, 5, len(bgf))])
        rf = RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced",
                                    random_state=seed, n_jobs=-1)
        aucs = []
        for tr, te in GroupKFold(n_splits=min(5, len(pos))).split(X, y, groups):
            if len(set(y[te])) < 2:
                continue
            rf.fit(X[tr], y[tr])
            aucs.append(roc_auc_score(y[te], rf.predict_proba(X[te])[:, 1]))
        rf.fit(X, y)
        log(f"gem ML model: {len(pos)} occurrences, cross-validated AUC "
            f"{np.mean(aucs):.3f}" if aucs else "gem ML model trained (too few folds for CV)")
        for i, key in enumerate(keys):
            progress(0.8 + 0.15 * i / len(keys), f"Data-driven gem model: tile {key}")
            t, scene, sl = self._read_tile(key, halo=16)
            usable = self._gem_usable(scene)
            ev = evidence_layers(scene.reflectance, usable, stats, self.config.resolution, None, scene.dem,
                                 self._cos_i(key, scene))
            core_ok = usable[sl] & self.aoi_mask(t.info)
            F = np.stack([ev[f][sl] for f in FEATURE_KEYS], -1).reshape(-1, len(FEATURE_KEYS))
            prob = np.zeros(core_ok.size, np.float32)
            idx = np.nonzero(core_ok.ravel())[0]
            if len(idx):
                prob[idx] = rf.predict_proba(F[idx])[:, 1]
            band = np.where(core_ok, np.round(prob.reshape(core_ok.shape) * 100) + 1, 0).astype(np.uint8)
            with rasterio.open(self.tile_dir(key) / "gems.tif") as src:
                data = src.read()[:6]
            write_raster(self.tile_dir(key) / "gems.tif", np.concatenate([data, band[None]]), t.info, nodata=0)
        return {"occurrences": len(pos), "cv_auc": round(float(np.mean(aucs)), 3) if aucs else None,
                "importance": dict(zip(FEATURE_KEYS, [round(float(v), 3) for v in rf.feature_importances_]))}


def replace_file(src: Path, dst: Path, attempts: int = 40) -> None:
    """os.replace with retries: on Windows the target may briefly be open in another thread/process."""
    for i in range(attempts):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(0.05 * (i + 1))


def _ring_area(geom: dict) -> float:
    total = 0.0
    for i, ring in enumerate(geom["coordinates"]):
        a = np.asarray(ring)
        s = 0.5 * abs(np.dot(a[:, 0], np.roll(a[:, 1], 1)) - np.dot(a[:, 1], np.roll(a[:, 0], 1)))
        total += s if i == 0 else -s
    return total


def _write_vrt(path: Path, region: Region, keys: Sequence[str], filename: str, count: int, dtype: str, nodata) -> None:
    """Write a GDAL VRT mosaicking per-tile files that share the region grid."""
    t = region.transform
    ts = region.config.tile_size
    srs = region.crs.to_wkt().replace("&", "&amp;").replace('"', "&quot;").replace("<", "&lt;")
    lines = [f'<VRTDataset rasterXSize="{region.width}" rasterYSize="{region.height}">',
             f"  <SRS>{srs}</SRS>",
             f"  <GeoTransform>{t.c}, {t.a}, {t.b}, {t.f}, {t.d}, {t.e}</GeoTransform>"]
    for b in range(1, count + 1):
        lines.append(f'  <VRTRasterBand dataType="{dtype}" band="{b}">')
        lines.append(f"    <NoDataValue>{nodata}</NoDataValue>")
        for k in keys:
            tile = region.tile(k)
            lines += [
                "    <SimpleSource>",
                f'      <SourceFilename relativeToVRT="1">tiles/{k}/{filename}</SourceFilename>',
                f"      <SourceBand>{b}</SourceBand>",
                f'      <SrcRect xOff="0" yOff="0" xSize="{ts}" ySize="{ts}"/>',
                f'      <DstRect xOff="{int(tile.window.col_off)}" yOff="{int(tile.window.row_off)}" xSize="{ts}" ySize="{ts}"/>',
                "    </SimpleSource>"]
        lines.append("  </VRTRasterBand>")
    lines.append("</VRTDataset>")
    path.write_text("\n".join(lines), encoding="utf-8")


def region_legend() -> list[dict]:
    return [{"id": c.id, "name": c.name, "color": c.color} for c in ALL_CLASSES.values() if c.id != CLOUD_CLASS]


__all__ = ["Region", "RegionConfig", "load_geojson_geometry", "region_legend", "hex_to_rgb"]


def _write_targets(out: Path, targets: list[dict]) -> None:
    import csv
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
              "properties": {k: v for k, v in t.items() if k not in ("x", "y")}} for t in targets]
    (out / "targets.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}), encoding="utf-8")
    cols = ["id", "lat", "lon", "type_label", "mean_score", "peak_score", "area_ha", "elevation_m", "lithology", "tile"]
    with open(out / "targets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in targets:
            w.writerow([t.get(c, "") for c in cols])


def _write_gem_targets(out: Path, targets: list[dict]) -> None:
    import csv
    feats = [{"type": "Feature", "geometry": {"type": "Point", "coordinates": [t["lon"], t["lat"]]},
              "properties": {k: v for k, v in t.items() if k not in ("x", "y")}} for t in targets]
    (out / "gem_targets.geojson").write_text(json.dumps({"type": "FeatureCollection", "features": feats}),
                                             encoding="utf-8")
    cols = ["id", "lat", "lon", "model_name", "gems", "mean_score", "peak_score", "area_ha", "elevation_m", "tile"]
    with open(out / "gem_targets.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for t in targets:
            w.writerow([t.get(c, "") for c in cols])


def _band_count(path: Path) -> int:
    with rasterio.open(path) as src:
        return src.count
