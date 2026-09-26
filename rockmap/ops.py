"""Operations helpers: installation self-check (`rockmap doctor`), backup and restore."""
from __future__ import annotations

import json
import os
import platform
import shutil
import socket
import sqlite3
import sys
import tempfile
import time
import zipfile
from contextlib import closing
from pathlib import Path

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"


def doctor(data: Path, offline: bool = False, port: int = 5000) -> int:
    """Check the installation; returns the number of failures (0 = healthy)."""
    results: list[tuple[str, str, str]] = []

    def add(status, what, detail=""):
        results.append((status, what, detail))

    v = sys.version_info
    add(OK if (3, 9) <= v[:2] <= (3, 13) else WARN, f"Python {platform.python_version()} ({platform.system()} "
        f"{platform.machine()})", "" if (3, 9) <= v[:2] <= (3, 13) else "tested with Python 3.9-3.13")
    for mod, why in (("numpy", ""), ("scipy", ""), ("sklearn", "Random Forest / SVM"), ("rasterio", "GeoTIFF / GDAL"),
                     ("torch", "CNN"), ("flask", "dashboard"), ("waitress", "production web server"),
                     ("PIL", "images"), ("matplotlib", "reports")):
        try:
            m = __import__(mod)
            try:
                from importlib.metadata import version
                ver = version({"sklearn": "scikit-learn", "PIL": "pillow"}.get(mod, mod))
            except Exception:  # noqa: BLE001
                ver = ""
            extra = ""
            if mod == "rasterio":
                extra = f", GDAL {m.__gdal_version__}"
            if mod == "torch":
                extra = ", GPU (CUDA) available" if m.cuda.is_available() else ", CPU only"
            add(OK, f"{mod} {ver}{extra}", why)
        except Exception as e:  # noqa: BLE001
            add(FAIL, f"{mod} missing or broken", f"{why}: {e} -> pip install -e .")
    try:
        data.mkdir(parents=True, exist_ok=True)
        probe = data / ".write_test"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()
        add(OK, f"data folder writable: {data.resolve()}")
    except OSError as e:
        add(FAIL, f"data folder not writable: {data}", str(e))
    free_gb = shutil.disk_usage(data if data.exists() else Path.cwd()).free / 1e9
    add(OK if free_gb > 10 else (WARN if free_gb > 2 else FAIL), f"free disk space {free_gb:,.1f} GB",
        "" if free_gb > 10 else "a 40 km region needs ~0.2 GB, all of Gilgit-Baltistan ~5 GB")
    db = data / "rockmap.db"
    if db.exists():
        try:
            with closing(sqlite3.connect(db)) as c:
                users = c.execute("SELECT COUNT(*) FROM users").fetchone()[0]
                regions = c.execute("SELECT COUNT(*) FROM regions").fetchone()[0]
                stuck = c.execute("SELECT COUNT(*) FROM jobs WHERE status='running' AND heartbeat < ?",
                                  (time.time() - 600,)).fetchone()[0]
                integrity = c.execute("PRAGMA quick_check").fetchone()[0]
            add(OK if integrity == "ok" else FAIL, f"database: {users} users, {regions} regions",
                "" if integrity == "ok" else f"integrity check: {integrity} - restore a backup")
            if stuck:
                add(WARN, f"{stuck} job(s) look stuck", "restart the server; they are re-queued automatically")
        except sqlite3.Error as e:
            add(FAIL, "database unreadable", str(e))
    else:
        add(OK, "database not created yet (it is created on first start)")
    try:
        with socket.socket() as s:
            s.bind(("127.0.0.1", port))
        add(OK, f"port {port} is free")
    except OSError:
        add(WARN, f"port {port} is in use", f"RockMap may already be running, or use: rockmap serve --port {port + 80}")
    if not offline:
        from .acquisition import DEM_BUCKET, S2_BUCKET, http_get
        for name, url in (("Sentinel-2 archive (AWS)", f"{S2_BUCKET}/?list-type=2&max-keys=1&prefix=sentinel-s2-l2a-cogs/43/S/DV/2024/9/"),
                          ("Copernicus DEM (AWS)", f"{DEM_BUCKET}/?list-type=2&max-keys=1")):
            try:
                http_get(url, timeout=20, retries=1)
                add(OK, f"internet: {name} reachable")
            except Exception as e:  # noqa: BLE001
                add(FAIL, f"internet: {name} NOT reachable", f"{e}. Needed only to download imagery; "
                    "behind a proxy set HTTPS_PROXY (and CURL_CA_BUNDLE / SSL_CERT_FILE for TLS inspection)")
    print("\nRockMap installation check\n" + "-" * 60)
    for status, what, detail in results:
        print(f"[{status}] {what}" + (f"\n        {detail}" if detail and status != OK else ""))
    fails = sum(1 for r in results if r[0] == FAIL)
    warns = sum(1 for r in results if r[0] == WARN)
    print("-" * 60 + f"\n{'All checks passed.' if not fails else f'{fails} problem(s) must be fixed.'}"
          + (f" {warns} warning(s)." if warns else ""))
    return fails


# ----------------------------------------------------------------------------- backup / restore
SKIP_DIRS = {"cache", "__pycache__"}
REDOWNLOADABLE = {"stack.tif", "dem.tif"}   # per-tile imagery / DEM: re-created by "Acquire"


def backup(data: Path, out: Path, full: bool = False) -> Path:
    """Zip the database (consistent online copy) and all data. ``full=False`` leaves out the bulky
    per-tile imagery and DEM (stack.tif, dem.tif), which "Acquire" can download again; maps,
    classifications, analytics, models, reports and field photos are always included."""
    data, out = Path(data), Path(out)
    if not (data / "rockmap.db").exists():
        raise FileNotFoundError(f"no RockMap database in {data}")
    with tempfile.TemporaryDirectory() as tmp:
        snap = Path(tmp) / "rockmap.db"
        # closing() matters: "with sqlite3.connect()" only commits, and Windows cannot delete an open file
        with closing(sqlite3.connect(data / "rockmap.db")) as src, closing(sqlite3.connect(snap)) as dst:
            src.backup(dst)
        out.parent.mkdir(parents=True, exist_ok=True)
        n = 0
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
            z.write(snap, "rockmap.db")
            z.writestr("rockmap-backup.json", json.dumps({"created": time.strftime("%Y-%m-%d %H:%M:%S"),
                                                          "full": full, "source": str(data.resolve())}))
            for p in sorted(data.rglob("*")):
                rel = p.relative_to(data)
                parts = rel.parts
                if p.is_dir() or rel.name.startswith("rockmap.db") or set(parts) & SKIP_DIRS:
                    continue
                if not full and (rel.name in REDOWNLOADABLE and "tiles" in parts or p.suffix == ".vrt"):
                    continue
                z.write(p, rel.as_posix())
                n += 1
    print(f"backup written: {out} ({out.stat().st_size / 1e6:,.1f} MB, {n + 1} files{', full' if full else ''})")
    return out


def restore(archive: Path, data: Path, force: bool = False) -> Path:
    data = Path(data)
    if (data / "rockmap.db").exists() and not force:
        raise FileExistsError(f"{data} already contains a RockMap database (use --force to overwrite)")
    data.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        if "rockmap.db" not in z.namelist():
            raise ValueError("not a RockMap backup")
        for name in z.namelist():
            target = (data / name).resolve()
            if not str(target).startswith(str(data.resolve())):   # zip-slip protection
                raise ValueError(f"unsafe path in archive: {name}")
        z.extractall(data)
    meta = json.loads((data / "rockmap-backup.json").read_text(encoding="utf-8"))
    (data / "rockmap-backup.json").unlink(missing_ok=True)
    print(f"restored backup of {meta['created']} into {data}. Start with: rockmap serve --data {data}")
    if not meta.get("full"):
        print("Maps, reports and models are restored. Imagery tiles were not in this backup: run 'Acquire' "
              "on a region before re-classifying or re-analysing it.")
    return data


def default_data_dir(arg: str | None) -> Path:
    return Path(arg or os.environ.get("ROCKMAP_DATA_DIR", "data"))
