"""Automatic satellite data acquisition.

* **Sentinel-2 L2A** cloud-optimised GeoTIFFs from the AWS Open Data bucket
  ``sentinel-cogs`` (no account needed). Scenes are found either through the Earth Search
  STAC API or, if that is unreachable, by listing the bucket by MGRS tile.
* **Copernicus DEM GLO-30** (30 m) from the AWS Open Data bucket ``copernicus-dem-30m``.

For each processing tile a *cloud-free composite* is built: the least-cloudy scenes of the
chosen season are read directly onto the tile grid (only the needed bytes are downloaded),
cloud / shadow pixels are removed with the Scene Classification Layer, each scene is
topographically corrected with the DEM, and the per-pixel median is taken. Snow-covered
observations are only used where no snow-free observation exists.
"""
from __future__ import annotations

import json
import math
import os
import re
import ssl
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform, transform_bounds

from .io import GeoInfo, write_raster
from .preprocessing import c_correction, illumination

S2_BUCKET = "https://sentinel-cogs.s3.us-west-2.amazonaws.com"
S2_PREFIX = "sentinel-s2-l2a-cogs"
DEM_BUCKET = "https://copernicus-dem-30m.s3.amazonaws.com"
DEFAULT_STAC_URL = "https://earth-search.aws.element84.com/v1"

# Earth Search asset keys for the canonical bands, and the matching file names in the bucket
S2_ASSETS = {"blue": "B02", "green": "B03", "red": "B04", "nir": "B08", "swir16": "B11", "swir22": "B12",
             "scl": "SCL"}
S2_BAND_ORDER = ("blue", "green", "red", "nir", "swir16", "swir22")
SCL_CLEAR = (2, 4, 5, 7)      # dark area, vegetation, bare soil, unclassified
SCL_SNOW, SCL_WATER = 11, 6

GDAL_REMOTE_ENV = dict(
    GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
    CPL_VSIL_CURL_ALLOWED_EXTENSIONS=".tif,.TIF,.tiff",
    GDAL_HTTP_MULTIRANGE="YES",
    GDAL_HTTP_MERGE_CONSECUTIVE_RANGES="YES",
    GDAL_HTTP_MAX_RETRY="5",
    GDAL_HTTP_RETRY_DELAY="2",
    GDAL_HTTP_TIMEOUT="120",
    VSI_CACHE="TRUE",
)

Log = Callable[[str], None]


def _ssl_context() -> ssl.SSLContext:
    cafile = os.environ.get("SSL_CERT_FILE") or os.environ.get("CURL_CA_BUNDLE") or os.environ.get("REQUESTS_CA_BUNDLE")
    return ssl.create_default_context(cafile=cafile) if cafile else ssl.create_default_context()


def http_get(url: str, data: Optional[bytes] = None, timeout: float = 60, retries: int = 4,
             headers: Optional[dict] = None) -> bytes:
    """GET/POST with retries and exponential back-off. Raises FileNotFoundError on 404."""
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, data=data, headers={"User-Agent": "rockmap/1.0", **(headers or {})})
            with urllib.request.urlopen(req, timeout=timeout, context=_ssl_context()) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code in (403, 404):
                raise FileNotFoundError(url) from e
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2 ** attempt)
    raise ConnectionError(f"Failed to fetch {url}: {last}")


# ---------------------------------------------------------------------------
# MGRS tiling (Sentinel-2 granules)
# ---------------------------------------------------------------------------
_LAT_BANDS = "CDEFGHJKLMNPQRSTUVWX"
_ROW_LETTERS = "ABCDEFGHJKLMNPQRSTUV"
_COL_SETS = ("ABCDEFGH", "JKLMNPQR", "STUVWXYZ")


def utm_epsg(lon: float, lat: float) -> int:
    zone = int((lon + 180) // 6) + 1
    return (32600 if lat >= 0 else 32700) + zone


def mgrs_id(zone: int, easting: float, northing: float, lat: float) -> str:
    """MGRS 100 km square id (e.g. ``43SDA``) of a UTM coordinate (northern hemisphere)."""
    e_idx, n_idx = int(easting // 100_000), int(northing // 100_000)
    col = _COL_SETS[(zone - 1) % 3][e_idx - 1]
    row = _ROW_LETTERS[(n_idx + (5 if zone % 2 == 0 else 0)) % 20]
    band = _LAT_BANDS[int((lat + 80) // 8)]
    return f"{zone:02d}{band}{col}{row}"


def mgrs_tiles_for_bounds(west: float, south: float, east: float, north: float) -> list[str]:
    """Sentinel-2 MGRS tile ids whose footprint may intersect a lon/lat box."""
    ids: set[str] = set()
    z0, z1 = int((west + 180) // 6) + 1, int((east + 180) // 6) + 1
    for zone in range(z0, z1 + 1):
        epsg = 32600 + zone
        lons = np.linspace(max(west, -180 + (zone - 1) * 6), min(east, -180 + zone * 6), 12)
        lats = np.linspace(south, north, 12)
        lo, la = np.meshgrid(lons, lats)
        xs, ys = transform("EPSG:4326", f"EPSG:{epsg}", lo.ravel().tolist(), la.ravel().tolist())
        # a Sentinel-2 granule spans its 100 km square plus 9.8 km to the east and south
        e0, e1 = int((min(xs) - 10_000) // 100_000), int(max(xs) // 100_000)
        n0, n1 = int(min(ys) // 100_000), int((max(ys) + 10_000) // 100_000)
        for ei in range(max(1, e0), min(8, e1) + 1):
            for ni in range(n0, n1 + 1):
                cx, cy = ei * 100_000 + 50_000, ni * 100_000 + 50_000
                clon, clat = transform(f"EPSG:{epsg}", "EPSG:4326", [cx], [cy])
                # squares straddling a latitude band boundary may be filed under either band
                for lat in (clat[0] - 0.5, clat[0], clat[0] + 0.5):
                    if -80 <= lat < 84:
                        ids.add(mgrs_id(zone, cx, cy, lat))
    return sorted(ids)


# ---------------------------------------------------------------------------
# Scene catalogue
# ---------------------------------------------------------------------------

@dataclass
class S2Item:
    id: str
    datetime: str
    cloud_cover: float
    sun_azimuth: float
    sun_elevation: float
    bbox: list
    epsg: int
    hrefs: dict
    scale: float = 1e-4
    offset: float = -0.1
    mgrs: str = ""

    @classmethod
    def from_stac(cls, d: dict) -> "S2Item":
        p = d["properties"]
        assets = d["assets"]
        hrefs = {k: assets[k]["href"] for k in S2_ASSETS if k in assets}
        rb = (assets.get("red", {}).get("raster:bands") or [{}])[0]
        epsg = p.get("proj:epsg") or int(str(p.get("proj:code", "EPSG:0")).split(":")[-1])
        mgrs = f"{p.get('mgrs:utm_zone', '')}{p.get('mgrs:latitude_band', '')}{p.get('mgrs:grid_square', '')}"
        offset = float(rb.get("offset", 0.0))
        if p.get("earthsearch:boa_offset_applied"):
            # Earth Search already removed BOA_ADD_OFFSET from the pixel values (DN = reflectance x 10000)
            # although raster:bands still advertises it; applying it again would darken every band.
            offset = 0.0
        return cls(d["id"], p["datetime"], float(p.get("eo:cloud_cover", 100.0)),
                   float(p.get("view:sun_azimuth", 135.0)), float(p.get("view:sun_elevation", 45.0)),
                   d.get("bbox", []), int(epsg or 0), hrefs, float(rb.get("scale", 1e-4)), offset, mgrs)

    @property
    def date(self) -> str:
        return self.datetime[:10]


@dataclass
class Season:
    """Acquisition window: years and months (late summer = least snow in the Karakoram)."""
    years: Sequence[int] = (2023, 2024, 2025)
    months: Sequence[int] = (7, 8, 9, 10)
    max_cloud: float = 30.0

    def ranges(self) -> list[tuple[str, str]]:
        out = []
        for y in self.years:
            for m in self.months:
                last = (np.datetime64(f"{y}-{m:02d}") + np.timedelta64(1, "M") - np.timedelta64(1, "D"))
                out.append((f"{y}-{m:02d}-01", str(last)))
        return out


class S2Catalog:
    """Finds Sentinel-2 L2A scenes via STAC, falling back to listing the S3 bucket."""

    def __init__(self, stac_url: Optional[str] = None, cache_dir: Optional[Path] = None,
                 backend: str = "auto", log: Log = print):
        self.stac_url = stac_url if stac_url is not None else os.environ.get("ROCKMAP_STAC_URL", DEFAULT_STAC_URL)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.backend = backend
        self.log = log
        self._stac_ok: Optional[bool] = None if backend == "auto" else backend == "stac"

    # -- STAC -------------------------------------------------------------------------
    def _stac_search(self, bbox, season: Season) -> list[S2Item]:
        items: list[S2Item] = []
        for start, end in season.ranges():
            body = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox), "limit": 200,
                    "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
                    "query": {"eo:cloud_cover": {"lt": season.max_cloud}}}
            url = f"{self.stac_url.rstrip('/')}/search"
            while url:
                d = json.loads(http_get(url, json.dumps(body).encode(), headers={"Content-Type": "application/json"}))
                items += [S2Item.from_stac(f) for f in d.get("features", [])]
                nxt = [lk for lk in d.get("links", []) if lk.get("rel") == "next"]
                url, body = (nxt[0]["href"], nxt[0].get("body", body)) if nxt else (None, None)
        return items

    # -- S3 listing -------------------------------------------------------------------
    def _list_prefixes(self, prefix: str) -> list[str]:
        out, token = [], None
        while True:
            q = f"{S2_BUCKET}/?list-type=2&delimiter=/&prefix={prefix}"
            if token:
                q += f"&continuation-token={urllib.request.quote(token)}"
            xml = http_get(q).decode()
            out += re.findall(r"<Prefix>([^<]+/)</Prefix>", xml)
            m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
            if not m:
                return [p for p in out if p != prefix]
            token = m.group(1)

    def _item_json(self, prefix: str) -> Optional[dict]:
        item_id = prefix.rstrip("/").split("/")[-1]
        cache = self.cache_dir / "items" / f"{item_id}.json" if self.cache_dir else None
        if cache and cache.exists():
            return json.loads(cache.read_text(encoding="utf-8"))
        try:
            d = json.loads(http_get(f"{S2_BUCKET}/{prefix}{item_id}.json"))
        except FileNotFoundError:
            return None
        if cache:
            cache.parent.mkdir(parents=True, exist_ok=True)
            cache.write_text(json.dumps(d), encoding="utf-8")
        return d

    def _s3_search(self, bbox, season: Season) -> list[S2Item]:
        tiles = mgrs_tiles_for_bounds(*bbox)
        prefixes = []
        for t in tiles:
            zone, band, sq = int(t[:2]), t[2], t[3:]
            for y in season.years:
                for m in season.months:
                    prefixes.append(f"{S2_PREFIX}/{zone}/{band}/{sq}/{y}/{m}/")
        with ThreadPoolExecutor(16) as ex:
            item_prefixes = [p for lst in ex.map(self._list_prefixes, prefixes) for p in lst]
            docs = list(ex.map(self._item_json, item_prefixes))
        items = []
        for d in docs:
            if not d:
                continue
            it = S2Item.from_stac(d)
            if it.cloud_cover < season.max_cloud and len(it.hrefs) == len(S2_ASSETS):
                items.append(it)
        return items

    def search(self, bbox: Sequence[float], season: Season) -> list[S2Item]:
        """Scenes intersecting ``bbox`` (lon/lat) in the season, sorted by cloud cover."""
        items: list[S2Item] = []
        if self._stac_ok is not False:
            try:
                items = self._stac_search(bbox, season)
                self._stac_ok = True
            except (ConnectionError, FileNotFoundError, json.JSONDecodeError) as e:
                if self.backend == "stac":
                    raise
                self.log(f"STAC API unavailable ({e}); listing the S3 bucket instead")
                self._stac_ok = False
        if self._stac_ok is False:
            items = self._s3_search(bbox, season)
        w, s, e, n = bbox
        items = [i for i in items if not i.bbox or (i.bbox[0] < e and i.bbox[2] > w and i.bbox[1] < n and i.bbox[3] > s)]
        # keep one item per (granule, date) - reprocessed duplicates share the date
        best: dict = {}
        for it in sorted(items, key=lambda i: i.cloud_cover):
            best.setdefault((it.mgrs or it.id.split("_")[1], it.date), it)
        return sorted(best.values(), key=lambda i: i.cloud_cover)


# ---------------------------------------------------------------------------
# Reading remote rasters onto a tile grid
# ---------------------------------------------------------------------------

def read_to_grid(url: str, grid: GeoInfo, resampling: Resampling = Resampling.bilinear,
                 band: int = 1, dtype="float32") -> np.ndarray:
    """Read one band of a (remote) raster reprojected onto ``grid``; missing areas are NaN / 0."""
    path = url if not url.startswith("http") else f"/vsicurl/{url}"
    with rasterio.Env(**GDAL_REMOTE_ENV):
        with rasterio.open(path) as src:
            nodata = src.nodata if src.nodata is not None else 0
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width, height=grid.height,
                           resampling=resampling, src_nodata=nodata, nodata=nodata) as vrt:
                data = vrt.read(band).astype(dtype)
    if np.issubdtype(np.dtype(dtype), np.floating):
        data[data == nodata] = np.nan
    return data


def copernicus_dem_urls(bounds_wgs84: Sequence[float]) -> list[str]:
    w, s, e, n = bounds_wgs84
    urls = []
    for lat in range(math.floor(s), math.ceil(n)):
        for lon in range(math.floor(w), math.ceil(e)):
            ns = f"N{lat:02d}" if lat >= 0 else f"S{-lat:02d}"
            ew = f"E{lon:03d}" if lon >= 0 else f"W{-lon:03d}"
            name = f"Copernicus_DSM_COG_10_{ns}_00_{ew}_00_DEM"
            urls.append(f"{DEM_BUCKET}/{name}/{name}.tif")
    return urls


def fetch_dem(grid: GeoInfo, out_path: Optional[Path] = None, log: Log = print) -> np.ndarray:
    """Copernicus GLO-30 elevation (m) on the tile grid."""
    bounds = _grid_wgs84(grid)
    dem = np.full((grid.height, grid.width), np.nan, np.float32)

    def one(url):
        try:
            return read_to_grid(url, grid, Resampling.bilinear)
        except rasterio.errors.RasterioIOError:
            return None  # tile absent (sea) or unreachable
    with ThreadPoolExecutor(4) as ex:
        for part in ex.map(one, copernicus_dem_urls(bounds)):
            if part is not None:
                dem = np.where(np.isnan(dem), part, dem)
    if out_path is not None:
        write_raster(out_path, dem[None], grid, nodata=np.nan, descriptions=["elevation_m"])
    return dem


def _grid_wgs84(grid: GeoInfo) -> tuple[float, float, float, float]:
    left, top = grid.transform @ (0, 0)
    right, bottom = grid.transform @ (grid.width, grid.height)
    return transform_bounds(grid.crs, "EPSG:4326", left, bottom, right, top, densify_pts=21)


# ---------------------------------------------------------------------------
# Composites
# ---------------------------------------------------------------------------

@dataclass
class CompositeReport:
    items: list = field(default_factory=list)
    dates: list = field(default_factory=list)
    clear_fraction: float = 0.0
    snow_fraction: float = 0.0
    sun_azimuth: float = 0.0
    sun_elevation: float = 0.0
    seconds: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


def read_item(item: S2Item, grid: GeoInfo, threads: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Surface reflectance (6, H, W) and SCL (H, W) of one scene on the tile grid."""
    keys = list(S2_BAND_ORDER) + ["scl"]

    def one(k):
        rs = Resampling.nearest if k == "scl" else Resampling.average
        return read_to_grid(item.hrefs[k], grid, rs, dtype="float32")
    with ThreadPoolExecutor(threads) as ex:
        arrays = list(ex.map(one, keys))
    dn = np.stack(arrays[:6])
    refl = dn * item.scale + item.offset
    scl = np.nan_to_num(arrays[6], nan=0).astype(np.uint8)
    refl[:, scl == 0] = np.nan
    return refl.astype(np.float32), scl


def build_composite(grid: GeoInfo, items: Sequence[S2Item], dem: Optional[np.ndarray] = None,
                    max_items: int = 6, target_clear: float = 0.97, topo_correct: bool = True,
                    log: Log = print, reader: Callable = read_item) -> tuple[np.ndarray, CompositeReport]:
    """Median cloud-free composite on ``grid``.

    Returns a (7, H, W) uint16 stack: 6 bands of reflectance x 10000 followed by an
    SCL-compatible code band (4 clear, 11 snow, 6 water, 0 no data).
    """
    t0 = time.time()
    rep = CompositeReport()
    h, w = grid.height, grid.width
    clear_obs: list[np.ndarray] = []
    snow_obs: list[np.ndarray] = []
    water_votes = np.zeros((h, w), np.int16)
    n_valid = np.zeros((h, w), np.int16)
    covered = np.zeros((h, w), bool)
    suns = []
    for item in items[: max_items * 3]:
        if len(rep.items) >= max_items or (len(rep.items) >= 2 and covered.mean() >= target_clear):
            break
        try:
            refl, scl = reader(item, grid)
        except Exception as e:  # noqa: BLE001 - skip unreadable scenes, keep going
            log(f"  skipped {item.id}: {e}")
            continue
        finite = np.all(np.isfinite(refl), axis=0)
        clear = finite & np.isin(scl, SCL_CLEAR + (SCL_WATER,))
        snow = finite & (scl == SCL_SNOW)
        if not (clear.any() or snow.any()):
            continue
        if topo_correct and dem is not None and np.isfinite(dem).any():
            cos_i = illumination(dem, grid.pixel_size[0], item.sun_azimuth, item.sun_elevation)
            refl = c_correction(refl, cos_i, item.sun_elevation, clear)
        c = np.where(clear[None], refl, np.nan)
        s = np.where(snow[None], refl, np.nan)
        clear_obs.append(c)
        snow_obs.append(s)
        water_votes += (scl == SCL_WATER) & finite
        n_valid += clear | snow
        covered |= clear
        suns.append((item.sun_azimuth, item.sun_elevation))
        rep.items.append(item.id)
        rep.dates.append(item.date)
        log(f"  {item.id} cloud {item.cloud_cover:.0f}% -> clear coverage {covered.mean() * 100:.1f}%")

    out = np.zeros((7, h, w), np.uint16)
    if not clear_obs:
        rep.seconds = time.time() - t0
        return out, rep
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)   # all-NaN slices -> NaN
        med = np.nanmedian(np.stack(clear_obs), axis=0)
        snow_med = np.nanmedian(np.stack(snow_obs), axis=0)
    has_clear = np.all(np.isfinite(med), axis=0)
    has_snow = ~has_clear & np.all(np.isfinite(snow_med), axis=0)
    refl = np.where(has_clear[None], med, np.where(has_snow[None], snow_med, np.nan))
    ok = np.all(np.isfinite(refl), axis=0)
    out[:6] = np.where(ok[None], np.clip(np.nan_to_num(refl) * 10000, 1, 65535), 0).astype(np.uint16)
    code = np.zeros((h, w), np.uint8)
    code[has_clear] = 4
    code[has_clear & (water_votes * 2 > n_valid)] = SCL_WATER
    code[has_snow] = SCL_SNOW
    out[6] = code
    rep.clear_fraction = float(has_clear.mean())
    rep.snow_fraction = float(has_snow.mean())
    rep.sun_azimuth = float(np.mean([s[0] for s in suns]))
    rep.sun_elevation = float(np.mean([s[1] for s in suns]))
    rep.seconds = round(time.time() - t0, 1)
    return out, rep


def write_composite(path: Path, stack: np.ndarray, grid: GeoInfo, report: CompositeReport) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    prof = dict(driver="GTiff", width=grid.width, height=grid.height, count=stack.shape[0], dtype="uint16",
                crs=grid.crs, transform=grid.transform, nodata=0, compress="deflate", predictor=2,
                tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(stack)
        for i, d in enumerate(["blue", "green", "red", "nir", "swir1", "swir2", "scl_code"], start=1):
            dst.set_band_description(i, d)
        dst.update_tags(**{k: json.dumps(v) for k, v in report.to_dict().items()})
    return path


def read_local_to_grid(paths: Iterable[str | Path], grid: GeoInfo, bands: int = 6,
                       resampling: Resampling = Resampling.bilinear) -> np.ndarray:
    """Mosaic user-supplied rasters (any CRS) onto ``grid``: first valid value wins."""
    out = np.full((bands, grid.height, grid.width), np.nan, np.float32)
    for p in paths:
        with rasterio.open(p) as src:
            nodata = src.nodata if src.nodata is not None else 0
            with WarpedVRT(src, crs=grid.crs, transform=grid.transform, width=grid.width,
                           height=grid.height, resampling=resampling, src_nodata=nodata, nodata=nodata) as vrt:
                data = vrt.read(list(range(1, min(bands, src.count) + 1))).astype(np.float32)
        data[data == nodata] = np.nan
        n = data.shape[0]
        out[:n] = np.where(np.isnan(out[:n]), data, out[:n])
    return out
