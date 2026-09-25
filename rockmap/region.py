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
        self.config = RegionConfig(**json.loads((self.folder / "region.json").read_text()))
        g = json.loads((self.folder / "grid.json").read_text())
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
        (folder / "region.json").write_text(json.dumps(asdict(config), indent=2))
        (folder / "grid.json").write_text(json.dumps(grid))
        (folder / "state.json").write_text(json.dumps({"tiles": {}, "log": []}))
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
        return json.loads((self.folder / "grid.json").read_text())["aoi_projected"]

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
        return json.loads(p.read_text()) if p.exists() else {"tiles": {}, "log": []}

    def _update_tile(self, key: str, **values) -> None:
        with self._lock:
            st = self.state()
            st["tiles"].setdefault(key, {}).update(values)
            tmp = self.folder / "state.json.tmp"
            tmp.write_text(json.dumps(st))
            tmp.replace(self.folder / "state.json")

    def _set(self, **values) -> None:
        with self._lock:
            st = self.state()
            st.update(values)
            tmp = self.folder / "state.json.tmp"
            tmp.write_text(json.dumps(st))
            tmp.replace(self.folder / "state.json")

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
        (d / "meta.json").write_text(json.dumps(meta, indent=2))
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
        products = {"rgb": (3, acquired), "falsecolor": (3, acquired), "hillshade": (1, acquired),
                    "surface": (1, acquired), "lithology": (1, classified), "confidence": (1, classified)}
        paths = {}
        for pi, (name, (count, keys)) in enumerate(products.items()):
            if not keys:
                continue
            path = out / f"{name}.tif"
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
                    rs = Resampling.nearest if name in ("lithology", "surface") else Resampling.average
                    dst.build_overviews(factors, rs)
                    dst.update_tags(ns="rio_overview", resampling=rs.name)
            paths[name] = path
        self._set(mosaic_built=time.strftime("%Y-%m-%d %H:%M:%S"), mosaic_version=int(time.time()))
        progress(1.0, "Mosaics built")
        return paths

    def _render_tile(self, key: str, name: str) -> np.ndarray:
        d = self.tile_dir(key)
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
        with open(out_path, "w") as fh:
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
    path.write_text("\n".join(lines))


def region_legend() -> list[dict]:
    return [{"id": c.id, "name": c.name, "color": c.color} for c in ALL_CLASSES.values() if c.id != CLOUD_CLASS]


__all__ = ["Region", "RegionConfig", "load_geojson_geometry", "region_legend", "hex_to_rgb"]
