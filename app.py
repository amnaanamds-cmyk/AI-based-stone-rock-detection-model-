"""Start the RockMap web application.

    python app.py                 # http://127.0.0.1:5000 (opens the browser)
    python app.py --port 8080     # any option of "rockmap serve"

The application loads the current trained model from models/trained/ (see models/README.md) and
picks up newly trained versions automatically. It never trains a model itself: training happens only
in the training module (python training/train.py).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from rockmap.cli import main  # noqa: E402

if __name__ == "__main__":
    main(["serve", "--open", *sys.argv[1:]])
