"""PDF report generation for regions and single-scene classifications."""
from __future__ import annotations

import json
import textwrap
import time
from pathlib import Path
from typing import Optional

import numpy as np

from . import __version__
from .config import ALL_CLASSES, CLASS_IDS, CLOUD_CLASS


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def _table(ax, rows, header, col_widths=None, fontsize=8.5):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="left", colLoc="left",
                 colWidths=col_widths)
    t.auto_set_font_size(False)
    t.set_fontsize(fontsize)
    t.scale(1, 1.35)
    for (r, _c), cell in t.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if r == 0:
            cell.set_facecolor("#efe9df")
            cell.set_text_props(weight="bold")
    return t


def region_report(region, out_path, model_meta: Optional[dict] = None, stats: Optional[dict] = None,
                  organisation: str = "", analytics: Optional[dict] = None,
                  validation: Optional[dict] = None, gems: Optional[dict] = None,
                  minerals: Optional[dict] = None) -> Path:
    """Multi-page PDF: map, area statistics, district table, model accuracy, data & method."""
    import rasterio
    from matplotlib.backends.backend_pdf import PdfPages
    from matplotlib.patches import Patch

    from .mapping import colorize
    plt = _plt()
    out_path = Path(out_path)
    stats = stats or region.statistics()
    st = region.state()
    cfg = region.config
    lith = region.folder / "mosaic" / "lithology.tif"
    hill = region.folder / "mosaic" / "hillshade.tif"

    with PdfPages(out_path) as pdf:
        # --- page 1: map -------------------------------------------------------------
        fig = plt.figure(figsize=(11.69, 8.27))  # A4 landscape
        fig.suptitle(f"Lithological map - {cfg.name}", fontsize=16, weight="bold", x=0.02, ha="left", y=0.97)
        sub = (f"RockMap v{__version__} - generated {time.strftime('%d %B %Y')}"
               + (f" - {organisation}" if organisation else ""))
        fig.text(0.02, 0.915, sub, fontsize=9, color="#555")
        ax = fig.add_axes([0.02, 0.05, 0.68, 0.84])
        ax.axis("off")
        if lith.exists():
            with rasterio.open(lith) as src:
                scale = max(1.0, max(src.width, src.height) / 2200)
                shape = (int(src.height / scale), int(src.width / scale))
                lab = src.read(1, out_shape=shape)
            base = None
            if hill.exists():
                with rasterio.open(hill) as src:
                    base = src.read(1, out_shape=shape)
            if base is not None:
                ax.imshow(np.ma.masked_equal(base, 0), cmap="gray", interpolation="bilinear")
            rgba = colorize(lab)
            rgba[..., 3] = np.where(lab > 0, 215, 0)
            ax.imshow(rgba, interpolation="nearest")
            present = [c for c in ALL_CLASSES if c != CLOUD_CLASS and (lab == c).any()]
            handles = [Patch(facecolor=ALL_CLASSES[c].color, edgecolor="#333", label=ALL_CLASSES[c].name)
                       for c in present]
            fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.72, 0.88), fontsize=9,
                       frameon=False, title="Legend", title_fontsize=10)
        else:
            ax.text(0.5, 0.5, "Mosaic not built yet", ha="center")
        s = region.summary()
        info = [f"Area of interest: {s['area_km2']:,.0f} km2 grid, {s['tiles']} tiles",
                f"Classified tiles: {s['classified']} / {s['tiles']}",
                f"Resolution: {cfg.resolution:g} m, CRS EPSG:{cfg.epsg}",
                f"Imagery: Sentinel-2 L2A, {min(cfg.years)}-{max(cfg.years)}, months {', '.join(map(str, cfg.months))}"
                if cfg.source == "sentinel2" else "Imagery: user-supplied scenes",
                f"Algorithm: {st.get('algorithm', '-')}"]
        fig.text(0.72, 0.30, "\n".join(info), fontsize=8.5, va="top", family="monospace")
        fig.text(0.72, 0.07, textwrap.fill(
            "Disclaimer: automated remote-sensing classification of surface lithology. Accuracy depends on "
            "the reference data used for training; verify in the field before engineering, mining or legal use.",
            60), fontsize=7.5, color="#666")
        pdf.savefig(fig)
        plt.close(fig)

        # --- page 2: executive summary ----------------------------------------------
        from .analytics import insights
        found = insights(stats, analytics, gems, minerals)
        if found:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Executive summary", fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
            y = 0.9
            for line in found:
                wrapped = textwrap.fill(line, 88)
                fig.text(0.06, y, "\u2022  " + wrapped.replace("\n", "\n    "), fontsize=10.5, va="top")
                y -= 0.03 * (wrapped.count("\n") + 1) + 0.02
            if validation and validation.get("metrics"):
                m = validation["metrics"]
                fig.text(0.06, y - 0.02, f"Field validation: {validation['n_compared']} observations, "
                         f"agreement {m['overall_accuracy'] * 100:.1f}%, kappa {m['kappa']:.2f}.",
                         fontsize=10.5, va="top", weight="bold")
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 2b: statistics -----------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Area statistics", fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
        rock = [r for r in stats["region"] if r["id"] in CLASS_IDS]
        masks = [r for r in stats["region"] if r["id"] not in CLASS_IDS]
        ax = fig.add_axes([0.32, 0.62, 0.6, 0.3])
        ax.barh([r["name"] for r in rock][::-1], [r["area_km2"] for r in rock][::-1],
                color=[r["color"] for r in rock][::-1], edgecolor="#333")
        ax.set_xlabel("km2")
        ax.spines[["top", "right"]].set_visible(False)
        ax2 = fig.add_axes([0.06, 0.18, 0.88, 0.38])
        rows = [[r["name"], f"{r['area_km2']:,.2f}", f"{r['percent']:.1f}" if r["percent"] is not None else "-"]
                for r in rock + masks]
        _table(ax2, rows, ["Class", "Area (km2)", "% of rock"], [0.55, 0.22, 0.2])
        pdf.savefig(fig)
        plt.close(fig)

        # --- page 3: districts -------------------------------------------------------
        if stats.get("districts"):
            fig = plt.figure(figsize=(11.69, 8.27))
            fig.suptitle("Lithology by district (km2)", fontsize=15, weight="bold", x=0.03, ha="left", y=0.97)
            ax = fig.add_axes([0.03, 0.05, 0.94, 0.85])
            short = [ALL_CLASSES[c].name.split(" /")[0].split(" (")[0] for c in CLASS_IDS]
            rows = []
            for d in stats["districts"]:
                by = {r["id"]: r["area_km2"] for r in d["stats"]}
                rows.append([d["name"]] + [f"{by.get(c, 0):,.1f}" for c in CLASS_IDS] + [f"{d['km2']:,.0f}"])
            _table(ax, rows, ["District"] + short + ["Total"], fontsize=7.5)
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 4: model -----------------------------------------------------------
        if model_meta:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Model accuracy (held-out spatial test blocks)", fontsize=14, weight="bold",
                         x=0.06, ha="left", y=0.97)
            ax = fig.add_axes([0.06, 0.72, 0.88, 0.2])
            rows = [[m["label"], f"{m['test_metrics']['overall_accuracy'] * 100:.2f}%",
                     f"{m['test_metrics']['kappa']:.3f}", f"{m['test_metrics']['macro_f1']:.3f}",
                     f"{m['test_metrics']['n_samples']:,}"] for m in model_meta["algorithms"].values()]
            _table(ax, rows, ["Algorithm", "Overall accuracy", "Kappa", "Macro F1", "Test pixels"])
            best = max(model_meta["algorithms"].values(), key=lambda m: m["test_metrics"]["kappa"])
            cm = np.asarray(best["test_metrics"]["confusion_matrix"], float)
            norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
            ax = fig.add_axes([0.25, 0.25, 0.6, 0.4])
            ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
            names = [ALL_CLASSES[c].name.split(" /")[0].split(" (")[0] for c in best["test_metrics"]["classes"]]
            ax.set_xticks(range(len(names)), names, rotation=40, ha="right", fontsize=8)
            ax.set_yticks(range(len(names)), names, fontsize=8)
            ax.set_title(f"Confusion matrix - {best['label']}", fontsize=10)
            for i in range(len(names)):
                for j in range(len(names)):
                    if cm[i, j]:
                        ax.text(j, i, f"{norm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                                color="white" if norm[i, j] > 0.5 else "black")
            fig.text(0.06, 0.12, textwrap.fill(
                f"Training pixels: {model_meta.get('n_train', 0):,}; validation {model_meta.get('n_val', 0):,}; "
                f"test {model_meta.get('n_test', 0):,}. Features: {', '.join(model_meta.get('feature_names', []))}.",
                110), fontsize=8)
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 4b: analytics -------------------------------------------------------
        if analytics:
            from .analytics import HAZARD_CLASSES
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Mineral alteration targets & landslide susceptibility", fontsize=14, weight="bold",
                         x=0.06, ha="left", y=0.97)
            tg = analytics.get("targets_top", [])[:15]
            ax = fig.add_axes([0.06, 0.5, 0.88, 0.42])
            if tg:
                rows = [[t["id"], f"{t['lat']:.5f}", f"{t['lon']:.5f}", t["type_label"].split(" (")[0],
                         f"{t['mean_score']:.0f}", f"{t['area_ha']:.1f}", f"{t.get('elevation_m', 0):.0f}"] for t in tg]
                _table(ax, rows, ["#", "Lat", "Lon", "Anomaly", "Score", "ha", "Elev m"],
                       [0.06, 0.16, 0.16, 0.26, 0.1, 0.1, 0.12], fontsize=8)
            else:
                ax.axis("off")
                ax.text(0, 0.9, "No alteration anomalies above the threshold.", fontsize=10)
            fig.text(0.06, 0.47, textwrap.fill(
                f"{analytics.get('targets_total', 0)} anomalies in total (full list in targets.csv). Scores are "
                "robust region-wide anomalies of Sentinel-2 clay (SWIR1/SWIR2), iron-oxide (red/blue) and "
                "ferrous ratios on snow-, vegetation- and shadow-free ground. They are screening targets for "
                "field checks, not proven mineralisation.", 100), fontsize=8, va="top", color="#444")
            hz = analytics.get("hazard_km2") or {}
            if sum(hz.values()):
                ax = fig.add_axes([0.25, 0.1, 0.65, 0.25])
                names = [HAZARD_CLASSES[int(k)][0] for k in sorted(hz)]
                ax.barh(names, [hz[k] for k in sorted(hz)], color=[HAZARD_CLASSES[int(k)][1] for k in sorted(hz)],
                        edgecolor="#333")
                ax.set_xlabel("km2")
                ax.set_title("Landslide / rockfall susceptibility (indicative)", fontsize=10)
                ax.spines[["top", "right"]].set_visible(False)
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 4c: gemstones -------------------------------------------------------
        if gems:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Gemstone prospectivity", fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
            ax = fig.add_axes([0.06, 0.78, 0.88, 0.15])
            rows = []
            for m in gems["models"]:
                v = m.get("validation") or {}
                rows.append([m["name"], m["gems"], str(m["targets"]),
                             f"AUC {v['auc']:.2f}, {v['top20'] * 100:.0f}% in top 20%" if v.get("n") else "-"])
            _table(ax, rows, ["Setting", "Gems", "Zones", "Validation (known localities)"],
                   [0.27, 0.36, 0.08, 0.29], fontsize=7.5)
            tg = gems.get("targets_top", [])[:18]
            ax = fig.add_axes([0.06, 0.3, 0.88, 0.44])
            if tg:
                rows = [[t["id"], t["model_name"].split(" ")[0], f"{t['lat']:.5f}", f"{t['lon']:.5f}",
                         f"{t['mean_score']:.0f}", f"{t['area_ha']:.1f}",
                         f"{t['elevation_m']:.0f}" if t.get("elevation_m") else "-"] for t in tg]
                _table(ax, rows, ["#", "Setting", "Lat", "Lon", "Score", "ha", "Elev m"],
                       [0.06, 0.18, 0.18, 0.18, 0.1, 0.1, 0.12], fontsize=8)
            else:
                ax.axis("off")
            ml = gems.get("ml")
            text = ("Gem crystals are far smaller than a 20 m pixel and cannot be detected directly. The maps rank the "
                    "host rocks and settings in which the gems of the Himalaya-Karakoram form: marble (ruby, spinel), "
                    "granitic pegmatite (aquamarine, topaz, tourmaline, garnet), pegmatite-mafic contacts (emerald, "
                    "beryl) and ultramafic rocks (peridot, nephrite), using Sentinel-2 spectral evidence on exposed "
                    "bedrock (slope >= 15 deg, snow, vegetation and sediments removed). Target zones are the top 1 % of "
                    "the region for each setting. They prioritise field visits and are not proven deposits.")
            if ml:
                text += (f" A data-driven Random Forest was trained on {ml['occurrences']} known localities"
                         + (f" (cross-validated AUC {ml['cv_auc']:.2f})." if ml.get("cv_auc") else "."))
            fig.text(0.06, 0.26, textwrap.fill(text, 100), fontsize=8.5, va="top", color="#333")
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 4d: structures & minerals -------------------------------------------
        if minerals:
            fig = plt.figure(figsize=(8.27, 11.69))
            fig.suptitle("Structures and mineral prospectivity", fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
            ax = fig.add_axes([0.06, 0.82, 0.88, 0.11])
            rows = []
            for m in minerals["models"]:
                v = m.get("validation") or {}
                rows.append([m["name"], m["commodities"], str(m["targets"]),
                             f"AUC {v['auc']:.2f}, {v['top20'] * 100:.0f}% in top 20%" if v.get("n") else "-"])
            _table(ax, rows, ["Model", "Commodities", "Zones", "Validation (known occurrences)"],
                   [0.3, 0.33, 0.08, 0.29], fontsize=7.5)
            ln = minerals.get("lineaments") or {}
            rose_bins = ln.get("rose") or []
            if rose_bins:   # rose diagram of lineament strikes (bidirectional)
                axr = fig.add_axes([0.06, 0.5, 0.3, 0.28], projection="polar")
                axr.set_theta_zero_location("N")
                axr.set_theta_direction(-1)
                w = np.radians(180.0 / len(rose_bins))
                for rb in rose_bins:
                    for off in (0, 180):
                        axr.bar(np.radians(rb["from"] + off) + w / 2, rb["km"], width=w, color="#444", edgecolor="white")
                axr.set_yticklabels([])
                axr.set_title(f"Lineament strikes\n{ln.get('segments', 0)} lines, {ln.get('total_km', 0):,.0f} km",
                              fontsize=8.5)
            tg = minerals.get("targets_top", [])[:16]
            ax = fig.add_axes([0.42, 0.44, 0.52, 0.35])
            if tg:
                rows = [[t["id"], t["model"], f"{t['lat']:.4f}", f"{t['lon']:.4f}", f"{t['mean_score']:.0f}",
                         f"{t['area_ha']:.1f}"] for t in tg]
                _table(ax, rows, ["#", "Model", "Lat", "Lon", "Score", "ha"], [0.1, 0.2, 0.22, 0.22, 0.12, 0.14],
                       fontsize=7.5)
            else:
                ax.axis("off")
            text = ("Lineaments are straight topographic features (fault- and fracture-controlled valleys, scarps and "
                    "ridges) extracted automatically from the Copernicus DEM with hillshades lit from four directions; "
                    "they are candidate structures for geological interpretation. Iron: ferric iron oxide and gossan "
                    "ratios. Copper: clay / sericite (Al-OH) with an iron-oxide cap on fractured ground. Quartz veins "
                    "(antimony, gold): lineament density and proximity in pale, iron-poor rock"
                    + ("; the ASTER thermal Quartz Index is included." if minerals.get("aster") else
                       ". Quartz and stibnite have no Sentinel-2 signature, so without ASTER thermal data this model "
                       "is structural.")
                    + " Clay and sulfate minerals cannot be told apart with Sentinel-2; hyperspectral data is needed for "
                    "that. Target zones are the top 1 % of the region per model and are not proven deposits.")
            ml = minerals.get("ml")
            if ml:
                text += (f" A data-driven Random Forest was trained on {ml['occurrences']} known occurrences"
                         + (f" (cross-validated AUC {ml['cv_auc']:.2f})." if ml.get("cv_auc") else "."))
            fig.text(0.06, 0.4, textwrap.fill(text, 100), fontsize=8.5, va="top", color="#333")
            pdf.savefig(fig)
            plt.close(fig)

        # --- page 5: method & data ---------------------------------------------------
        fig = plt.figure(figsize=(8.27, 11.69))
        fig.suptitle("Data and method", fontsize=15, weight="bold", x=0.06, ha="left", y=0.97)
        dates = sorted({d for t in st["tiles"].values() for d in t.get("dates", [])})
        text = [
            "Imagery: Copernicus Sentinel-2 Level-2A surface reflectance (AWS Open Data, sentinel-cogs), "
            f"{len(dates)} acquisition dates" + (f" between {dates[0]} and {dates[-1]}." if dates else "."),
            "Elevation: Copernicus DEM GLO-30 (AWS Open Data).",
            "Pre-processing: cloud / shadow removal with the Scene Classification Layer, topographic "
            "C-correction, per-pixel median composite of the least-cloudy scenes; snow observations are used only "
            "where no snow-free observation exists.",
            "Masks: snow / glacier (NDSI), water, dense vegetation (NDVI) and deep terrain shadow are mapped as "
            "separate non-rock classes.",
            "Features: 6 reflectance bands, 7 spectral indices (clay, carbonate, iron-oxide, ferrous, NDVI, "
            "brightness, bare-rock) and 6 terrain features (elevation, slope, aspect, hillshade, roughness).",
            "Classifier: fully convolutional CNN (9 x 9 px context) benchmarked against Random Forest and SVM; "
            "spatially blocked train / validation / test split.",
            "Contains modified Copernicus Sentinel data and Copernicus DEM (ESA / European Union).",
        ]
        y = 0.9
        for para in text:
            wrapped = textwrap.fill(para, 95)
            fig.text(0.06, y, wrapped, fontsize=9, va="top")
            y -= 0.025 * (wrapped.count("\n") + 1) + 0.015
        pdf.savefig(fig)
        plt.close(fig)
    return out_path


def load_meta(model_dir) -> Optional[dict]:
    p = Path(model_dir) / "meta.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None
