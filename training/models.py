"""List, inspect and switch trained model versions.

    python training/models.py list               # all versions, * = used by the application
    python training/models.py info model_v2      # details and evaluation of one version
    python training/models.py use model_v1       # switch the application to another version (rollback)
    python training/models.py delete model_v1    # remove an old version (not the current one)
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rockmap import model_store                        # noqa: E402


def list_versions(root: Path) -> None:
    names = model_store.versions(root)
    if not names:
        print(f"No trained models in {root}. Train one: python training/train.py")
        return
    cur = model_store.current_version(root)
    print(f"Models in {root}  (* = used by the application)\n")
    print(f"   {'version':10s} {'trained':19s} {'model':5s} {'accuracy':>8s} {'kappa':>6s}  data")
    for n in names:
        m = model_store.read_meta(n, root)
        algo = m.get("app_algorithm") or max(m["algorithms"], key=lambda k: m["algorithms"][k]["test_metrics"]["kappa"])
        t = m["algorithms"][algo]["test_metrics"]
        data = ", ".join(d["name"] for d in m.get("dataset", [])) or Path(m.get("scene", "?")).name
        if m.get("sample_data"):
            data += "  [SAMPLE DATA]"
        print(f" {'*' if n == cur else ' '} {n:10s} {m['created']:19s} {algo:5s} {t['overall_accuracy'] * 100:7.1f}% "
              f"{t['kappa']:6.3f}  {data}")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Manage trained model versions")
    ap.add_argument("command", choices=["list", "info", "use", "delete"])
    ap.add_argument("version", nargs="?")
    ap.add_argument("--models-dir", help="models folder (default: models/trained)")
    a = ap.parse_args(argv)
    root = Path(a.models_dir) if a.models_dir else model_store.models_dir()
    if a.command == "list":
        return list_versions(root)
    if not a.version:
        sys.exit(f"give a version, e.g. python training/models.py {a.command} model_v1")
    try:
        if a.command == "info":
            ev = root / a.version / "evaluation.txt"
            if not ev.exists():
                from training.evaluate import write_report
                write_report(root / a.version)
            print(ev.read_text(encoding="utf-8"))
        elif a.command == "use":
            model_store.set_current(a.version, root)
            print(f"{a.version} is now the current model; the application uses it from its next request.")
        else:
            model_store.delete_version(a.version, root)
            print(f"deleted {a.version}")
    except (ValueError, FileNotFoundError) as e:
        sys.exit(str(e))


if __name__ == "__main__":
    main()
