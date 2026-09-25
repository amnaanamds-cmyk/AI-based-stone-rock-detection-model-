# Deployment and administration

## 1. Hardware
* **Minimum:** 4 CPU cores, 16 GB RAM, 50 GB free disk, broadband internet (for imagery download).
* **Recommended for all of Gilgit-Baltistan:** 8+ cores, 32 GB RAM, 100 GB SSD. An NVIDIA GPU speeds up CNN training and classification. It is used automatically when PyTorch has CUDA support.
* Runs on Linux, Windows and macOS.

## 2. Installation options

### A. Docker Compose (recommended for servers)
```bash
docker compose up -d --build
docker compose logs web | grep "First start"     # initial admin password
```
This runs two containers that share the `rockmap-data` volume:
* **web** serves the dashboard and API on port 5000.
* **worker** runs the processing jobs, so the website stays responsive during long jobs.

### B. Python on a workstation
```bash
python -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -e .
rockmap serve --host 0.0.0.0 --port 5000            # web server + 1 background worker thread
```
To run a separate worker process: start `rockmap serve --workers 0`, then run `rockmap worker` in a second terminal or as a service.

### systemd service (Linux)
```ini
# /etc/systemd/system/rockmap.service
[Unit]
Description=RockMap geological mapping platform
After=network-online.target

[Service]
User=rockmap
Environment=ROCKMAP_DATA_DIR=/srv/rockmap
Environment=ROCKMAP_SECURE_COOKIES=1
ExecStart=/opt/rockmap/.venv/bin/rockmap serve --host 127.0.0.1 --port 5000
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

## 3. HTTPS with a reverse proxy (nginx)
```nginx
server {
    listen 443 ssl;
    server_name rockmap.example.pk;
    ssl_certificate     /etc/letsencrypt/live/rockmap.example.pk/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/rockmap.example.pk/privkey.pem;
    client_max_body_size 4g;              # scene and map uploads
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Forwarded-For $remote_addr;
        proxy_read_timeout 300;
    }
}
```
Set `ROCKMAP_SECURE_COOKIES=1` when serving over HTTPS.

## 4. Configuration (environment variables)
| Variable | Default | Meaning |
|---|---|---|
| `ROCKMAP_DATA_DIR` | `./data` | All data: database, scenes, regions, models, logs |
| `ROCKMAP_ADMIN_PASSWORD` | random | Password for the `admin` user created on first start |
| `ROCKMAP_ADMIN_USER` | `admin` | Name of that first user |
| `ROCKMAP_SECRET_KEY` | generated | Session signing key (otherwise stored in `DATA_DIR/secret_key`) |
| `ROCKMAP_SECURE_COOKIES` | off | `1` = cookies only over HTTPS |
| `ROCKMAP_WORKERS` | `1` | Background job threads inside the web server (`0` = use `rockmap worker`) |
| `ROCKMAP_MAX_UPLOAD_MB` | `4096` | Maximum upload size |
| `ROCKMAP_ORGANISATION` | – | Organisation name shown in the footer and on PDF reports |
| `ROCKMAP_STAC_URL` | Earth Search | STAC API for Sentinel-2 search; if unreachable, RockMap lists the S3 bucket instead |
| `CURL_CA_BUNDLE` / `SSL_CERT_FILE` | system | CA bundle, needed behind TLS-inspecting proxies |
| `HTTPS_PROXY` | – | Outbound proxy for downloads |

## 5. Users and security
* **Roles:** *viewer* can view maps and download products. *analyst* can also upload data, draw training areas and run jobs. *admin* can also manage users, delete data, use server-side files and read the audit log.
* Manage users under **Admin ▸ Users**, or from the command line: `rockmap create-user ali --role analyst`.
* Passwords are stored as salted hashes (werkzeug / scrypt). After 5 failed sign-ins, the account is locked for 5 minutes for that IP address. Changing a password signs out that user's other sessions.
* Every form is protected against CSRF. Every significant action is recorded in **Admin ▸ Audit log**.
* File downloads are restricted to the data sub-folders, and path traversal is blocked.

## 6. Backups
Back up the whole `ROCKMAP_DATA_DIR`. For a consistent copy of the database while the server runs, use `sqlite3 rockmap.db ".backup rockmap-backup.db"`. You can delete region `cache/` folders and `tiles/*/stack.tif` to save space: tiles can be re-acquired, but then they must be classified again.

## 7. REST API
Authenticate with a personal token (**Profile ▸ Create token**): `Authorization: Bearer rmk_...`

| Method & path | Purpose |
|---|---|
| `GET /api/me` | Current user |
| `GET /api/regions` · `GET /api/regions/<id>` | Regions with progress summary |
| `GET /api/regions/<id>/tiles` | Tile grid (GeoJSON) with status |
| `GET /api/regions/<id>/query?lat=..&lon=..` | Rock type, confidence, elevation and reflectance at a point |
| `GET /api/regions/<id>/stats` | Area statistics (region and districts) |
| `GET/POST /api/regions/<id>/annotations` · `DELETE …/annotations/<aid>` | Training areas (`{"class_id": 4, "geometry": {GeoJSON Polygon}}`) |
| `POST /api/regions/<id>/jobs` | Start a job: `{"stage": "acquire"|"train"|"classify"|"products"|"pipeline", ...}` |
| `GET /api/jobs` · `GET /api/jobs/<id>` · `POST /api/jobs/<id>/cancel` | Job status, log tail, cancel |
| `GET /tiles/<region>/<layer>/{z}/{x}/{y}.png` | XYZ map tiles (`lithology`, `confidence`, `rgb`, `falsecolor`, `hillshade`), usable in QGIS as an XYZ layer |
| `GET /api/classes` · `/api/models` · `/api/scenes` | Reference lists |
| `GET /healthz` | Health check (no authentication) |

Example:
```bash
TOKEN=rmk_...
curl -H "Authorization: Bearer $TOKEN" "https://rockmap.example.pk/api/regions/1/query?lat=35.92&lon=74.31"
curl -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
     -d '{"stage": "classify", "model_id": 3}' https://rockmap.example.pk/api/regions/1/jobs
```

## 8. Operations
* **Health:** `GET /healthz` returns the status and the number of queued jobs.
* **Logs:** the server console, plus one log file per job in `DATA_DIR/logs/`, which you can also read on the job page.
* **Interrupted jobs:** a job whose worker stops sending heartbeats for 3 minutes is re-queued, up to 3 attempts. Region jobs skip tiles that are already done.
* **Upgrades:** the database schema is migrated automatically on start.
