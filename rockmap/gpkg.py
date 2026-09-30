"""Minimal OGC GeoPackage writer (points, lines, polygons in WGS84) - no GDAL / Fiona needed.

A GeoPackage is a SQLite database that ArcGIS Pro, QGIS and most GIS software open directly. RockMap
bundles every vector product of a region (targets, lineaments, occurrences, lithology polygons,
observations) into one ``.gpkg`` so clients get a single geodatabase instead of a folder of files.
"""
from __future__ import annotations

import sqlite3
import struct
from contextlib import closing
from pathlib import Path
from typing import Iterable

WGS84_WKT = ('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],'
             'PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433],AUTHORITY["EPSG","4326"]]')
_WKB = {"Point": 1, "LineString": 2, "Polygon": 3, "MultiPoint": 4, "MultiLineString": 5, "MultiPolygon": 6}
_GPKG_TYPE = {"Point": "POINT", "LineString": "LINESTRING", "Polygon": "POLYGON", "MultiPoint": "MULTIPOINT",
              "MultiLineString": "MULTILINESTRING", "MultiPolygon": "MULTIPOLYGON"}


def _wkb(geom: dict) -> bytes:
    t, c = geom["type"], geom["coordinates"]

    def pts(seq):
        return struct.pack("<I", len(seq)) + b"".join(struct.pack("<dd", float(p[0]), float(p[1])) for p in seq)

    head = struct.pack("<BI", 1, _WKB[t])
    if t == "Point":
        return head + struct.pack("<dd", float(c[0]), float(c[1]))
    if t == "LineString":
        return head + pts(c)
    if t == "Polygon":
        return head + struct.pack("<I", len(c)) + b"".join(pts(r) for r in c)
    parts = {"MultiPoint": "Point", "MultiLineString": "LineString", "MultiPolygon": "Polygon"}[t]
    return head + struct.pack("<I", len(c)) + b"".join(_wkb({"type": parts, "coordinates": p}) for p in c)


def _gpkg_geom(geom: dict) -> tuple[bytes, tuple]:
    xy = _coords(geom)
    xs, ys = [p[0] for p in xy], [p[1] for p in xy]
    env = (min(xs), max(xs), min(ys), max(ys))
    # header: magic, version 0, flags (little endian, envelope xy = 1), srs id
    return b"GP" + bytes([0, 0b00000011]) + struct.pack("<i", 4326) + struct.pack("<4d", *env) + _wkb(geom), env


def _coords(geom: dict) -> list:
    out = []

    def walk(c):
        if isinstance(c[0], (int, float)):
            out.append(c)
        else:
            for x in c:
                walk(x)
    walk(geom["coordinates"])
    return out


def _sql_type(values: list) -> str:
    vals = [v for v in values if v is not None]
    if vals and all(isinstance(v, bool) or isinstance(v, int) for v in vals):
        return "INTEGER"
    if vals and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
        return "REAL"
    return "TEXT"


def write_geopackage(path: str | Path, layers: dict[str, dict]) -> Path:
    """Write ``{layer_name: GeoJSON FeatureCollection}`` (EPSG:4326) to a new GeoPackage.

    Each layer must contain a single geometry type family; Polygon and MultiPolygon are stored as
    MULTIPOLYGON (and likewise for lines / points) when mixed. Empty layers are skipped.
    """
    path = Path(path)
    path.unlink(missing_ok=True)
    with closing(sqlite3.connect(path)) as db:
        db.execute("PRAGMA application_id = 1196444487")      # 'GPKG'
        db.execute("PRAGMA user_version = 10300")
        db.executescript("""
        CREATE TABLE gpkg_spatial_ref_sys (srs_name TEXT NOT NULL, srs_id INTEGER PRIMARY KEY,
            organization TEXT NOT NULL, organization_coordsys_id INTEGER NOT NULL, definition TEXT NOT NULL,
            description TEXT);
        CREATE TABLE gpkg_contents (table_name TEXT NOT NULL PRIMARY KEY, data_type TEXT NOT NULL,
            identifier TEXT UNIQUE, description TEXT DEFAULT '', last_change DATETIME NOT NULL DEFAULT
            (strftime('%Y-%m-%dT%H:%M:%fZ','now')), min_x DOUBLE, min_y DOUBLE, max_x DOUBLE, max_y DOUBLE,
            srs_id INTEGER, CONSTRAINT fk_gc_r_srs_id FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys(srs_id));
        CREATE TABLE gpkg_geometry_columns (table_name TEXT NOT NULL, column_name TEXT NOT NULL,
            geometry_type_name TEXT NOT NULL, srs_id INTEGER NOT NULL, z TINYINT NOT NULL, m TINYINT NOT NULL,
            CONSTRAINT pk_geom_cols PRIMARY KEY (table_name, column_name),
            CONSTRAINT fk_gc_tn FOREIGN KEY (table_name) REFERENCES gpkg_contents(table_name),
            CONSTRAINT fk_gc_srs FOREIGN KEY (srs_id) REFERENCES gpkg_spatial_ref_sys (srs_id));
        """)
        db.executemany("INSERT INTO gpkg_spatial_ref_sys VALUES (?,?,?,?,?,?)", [
            ("Undefined cartesian SRS", -1, "NONE", -1, "undefined", None),
            ("Undefined geographic SRS", 0, "NONE", 0, "undefined", None),
            ("WGS 84 geodetic", 4326, "EPSG", 4326, WGS84_WKT, "longitude/latitude coordinates in decimal degrees")])
        for name, fc in layers.items():
            feats = [f for f in (fc or {}).get("features", []) if f.get("geometry")]
            if not feats:
                continue
            types = {f["geometry"]["type"] for f in feats}
            base = {t.replace("Multi", "") for t in types}
            if len(base) != 1:
                raise ValueError(f"layer {name}: mixed geometry types {sorted(types)}")
            gtype = types.pop() if len(types) == 1 else "Multi" + base.pop()
            props = sorted({k for f in feats for k in (f.get("properties") or {})})
            cols = {k: _sql_type([(f.get("properties") or {}).get(k) for f in feats]) for k in props}
            safe = {k: "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in k) or "field" for k in props}
            coldefs = ", ".join(f'"{safe[k]}" {cols[k]}' for k in props)
            db.execute(f'CREATE TABLE "{name}" (fid INTEGER PRIMARY KEY AUTOINCREMENT, geom {_GPKG_TYPE[gtype]}'
                       + (f", {coldefs}" if coldefs else "") + ")")
            env = [180.0, -180.0, 90.0, -90.0]
            rows = []
            for f in feats:
                g = f["geometry"]
                if g["type"] != gtype:          # promote single part to multi part
                    g = {"type": gtype, "coordinates": [g["coordinates"]]}
                blob, e = _gpkg_geom(g)
                env = [min(env[0], e[0]), max(env[1], e[1]), min(env[2], e[2]), max(env[3], e[3])]
                p = f.get("properties") or {}
                vals = []
                for k in props:
                    v = p.get(k)
                    vals.append(v if v is None or isinstance(v, (int, float, str)) else str(v))
                rows.append([blob] + vals)
            ph = ", ".join("?" * (1 + len(props)))
            names = ", ".join(["geom"] + [f'"{safe[k]}"' for k in props])
            db.executemany(f'INSERT INTO "{name}" ({names}) VALUES ({ph})', rows)
            db.execute("INSERT INTO gpkg_contents (table_name, data_type, identifier, min_x, min_y, max_x, max_y, "
                       "srs_id) VALUES (?, 'features', ?, ?, ?, ?, ?, 4326)", (name, name, env[0], env[2], env[1], env[3]))
            db.execute("INSERT INTO gpkg_geometry_columns VALUES (?, 'geom', ?, 4326, 0, 0)", (name, _GPKG_TYPE[gtype]))
        db.commit()
    return path


def layer_names(path: str | Path) -> Iterable[str]:
    with closing(sqlite3.connect(path)) as db:
        return [r[0] for r in db.execute("SELECT table_name FROM gpkg_contents ORDER BY table_name")]
