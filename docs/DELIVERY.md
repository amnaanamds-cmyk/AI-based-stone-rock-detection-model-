# Delivering RockMap to a customer

This is the checklist for installing RockMap at a customer site and handing it over.

## 1. Choose the installation type

| Customer | Recommended setup |
|---|---|
| One geologist or a small office, Windows PC | **Windows desktop:** `run.bat` (§2) |
| Team or department, several users, phones in the field | **Server:** Docker Compose on a Linux server with HTTPS (§3) |
| No internet at the site | Either type. Acquire imagery on a connected machine, then move the data folder (§6) |

Minimum hardware: 4 CPU cores, 16 GB RAM and 50 GB of free disk. For all of Gilgit-Baltistan, use 8+ cores, 32 GB RAM and 100 GB of SSD.

## 2. Windows desktop installation (about 15 minutes)
1. Install **Python 3.11 (64-bit)** from python.org and tick **"Add python.exe to PATH"**.
2. Copy the RockMap folder to the PC, for example to `C:\RockMap`, and double-click **`run.bat`**. The first run installs everything and builds a demo (about 5–10 minutes).
3. The browser opens http://127.0.0.1:5000. Sign in as **admin / rockmap-demo**. You are then **required to set a new password**.
4. Run the self-check: open a Command Prompt in the folder and run `.venv\Scripts\rockmap doctor`. Every line should say `OK`.
5. Create a desktop shortcut to `run.bat serve` for daily use.

## 3. Server installation
```bash
git clone <repository> rockmap && cd rockmap
export ROCKMAP_ADMIN_PASSWORD='<strong initial password>'   # optional; otherwise one is generated
docker compose up -d --build
docker compose logs web | grep "First start"                 # generated password, if not set above
docker compose exec web rockmap doctor
```
Then put the server behind HTTPS ([DEPLOYMENT.md §3](DEPLOYMENT.md)) and set `ROCKMAP_SECURE_COOKIES=1`. HTTPS is also needed for GPS in the phone field app.

## 4. Acceptance test with the customer
Go through these together and tick them off:

- [ ] `rockmap doctor` shows **All checks passed**, including the internet checks.
- [ ] Sign in, change the admin password, and create one account per person (**Admin ▸ Users**) with the right role.
- [ ] Open the demo region: toggle the lithology, hazard and alteration layers, click the map, and open the PDF report.
- [ ] **Regions ▸ New region ▸ Gilgit city (40 × 40 km) ▸ Run everything.** Imagery appears tile by tile (5–10 min), followed by key findings, targets and hazard.
- [ ] Name 3–4 spectral units, then **Create rock map from units**. The lithology layer appears.
- [ ] Record a field observation from a phone. It appears on the map.
- [ ] Create a public share link and open it in a private browser window.
- [ ] Run `rockmap backup` and confirm the zip file is created.

## 5. What to agree with the customer before production mapping
* **Reference geology.** The rock map is only as good as its training data. Agree who provides the digitised geological map(s) or field points, and name the units with a geologist using `examples/gb_geology_mapping.json`.
* **Official boundaries.** The built-in Gilgit-Baltistan outline is approximate. Obtain the official region and district boundaries (GeoJSON).
* **Scope of the results.** Alteration targets and landslide susceptibility are *screening* products for planning field work. They are not proven deposits or engineering assessments. This wording is already in the app and the reports.

## 6. Operating it
| Task | How |
|---|---|
| Start / stop | `run.bat serve` / Ctrl + C, or `docker compose up -d` / `docker compose down` |
| Health check | `rockmap doctor`, or `GET /healthz` for monitoring |
| Users | **Admin ▸ Users**. New and reset passwords must be changed at first sign-in |
| Lost admin password | `rockmap create-user admin --reset` |
| Backup (daily recommended) | `rockmap backup --out D:\Backups\rockmap-%DATE%.zip`. Add `--full` to include the imagery tiles |
| Restore | `rockmap restore backup.zip --data <new folder>`, then `rockmap serve --data <new folder>` |
| Audit trail | **Admin ▸ Audit log**: sign-ins, uploads, jobs, deletions |
| Update to a new version | Back up, `git pull` (or replace the folder but keep `data/`), then `run.bat`. The database upgrades itself |
| Offline site | Acquire regions on a connected machine, then copy `data/` (or use backup/restore) |
| Logs | Server console, plus one file per job in `data/logs/` (also visible on the job page) |

## 7. Quality evidence you can show the customer
* 62 automated tests: processing, AI models, region engine, web platform, security, and a crawl of every page. They run on **Windows, macOS and Linux** at every change (GitHub Actions), including the one-command installers.
* Real-data verification on Sentinel-2 imagery of Gilgit: acquisition, masks, targets and hazard.
* Security: roles, CSRF protection, login lock-out, forced password change, hashed passwords and API tokens, audit log, path-traversal protection.

## 8. Before selling
* **Licence.** Add a LICENSE file that states the terms of use you choose (proprietary or open source). Third-party components are open source: Leaflet (BSD-2), PyTorch (BSD), scikit-learn (BSD), rasterio (BSD), GDAL (MIT/X), Flask (BSD).
* **Data credits.** Reports and maps must keep "Contains modified Copernicus Sentinel data" (the free Copernicus licence requires this). It is already printed on maps and reports.
