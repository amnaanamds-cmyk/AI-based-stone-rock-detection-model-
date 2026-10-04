"""All training settings in one place.

Change a value here and run ``python training/train.py`` again - or override it for one run on the
command line (``python training/train.py --epochs 50``). You never need to edit the other files to
train on new data.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:          # lets "python training/train.py" import the rockmap package
    sys.path.insert(0, str(PROJECT_ROOT))

from rockmap import model_store                 # noqa: E402
from rockmap.config import ROCK_CLASSES          # noqa: E402

# --------------------------------------------------------------------------- paths
# Your dataset: one sub-folder per study area (see dataset/README.md).
DATA_DIR = PROJECT_ROOT / "dataset" / "raw"
# Cache of preprocessed training samples (safe to delete; rebuilt automatically when data changes).
PROCESSED_DIR = PROJECT_ROOT / "dataset" / "processed"
# Every training run is saved here as model_v1, model_v2, ... (see models/README.md).
MODELS_DIR = model_store.models_dir()

# --------------------------------------------------------------------------- classes
# The lithology classes are defined ONCE for the whole system in rockmap/config.py (ROCK_CLASSES),
# because the application, maps, legends and reports use them too. Label values in your dataset must
# be these ids (0 = unlabelled). To add a class, add a RockClass line there - see training/README.md.
CLASS_NAMES = {c.id: c.name for c in ROCK_CLASSES}

# --------------------------------------------------------------------------- model
# Models trained in every run. "cnn" = convolutional neural network (PyTorch),
# "rf" = Random Forest, "svm" = Support Vector Machine (scikit-learn benchmarks).
MODEL_TYPES = ["cnn", "rf", "svm"]
# Which of them the application uses for prediction: "best" = highest kappa on the held-out test
# blocks, or force one of MODEL_TYPES.
APP_MODEL_TYPE = "best"

# The CNN looks at a PATCH_SIZE x PATCH_SIZE pixel neighbourhood (20 m pixels -> 180 m). This is the
# "image size" of this project: input scenes can be any size, they are processed patch by patch.
# It is fixed by the network architecture (rockmap/models/cnn.py, LithoCNN.receptive_field).
PATCH_SIZE = 9
CNN_WIDTH = 64              # filters per convolution layer (more = bigger model, slower)
EPOCHS = 30                 # passes over the training samples
BATCH_SIZE = 256
LEARNING_RATE = 0.002

# --------------------------------------------------------------------------- data sampling
SAMPLES_PER_CLASS = 4000    # training pixels drawn per class (validation / test get fewer)
# Train / validation / test shares. The split is made by square BLOCKS of pixels, not single pixels,
# so test pixels are never right next to training pixels (honest accuracy for maps).
SPLIT = {"train": 0.60, "val": 0.15, "test": 0.25}
BLOCK_SIZE = 32             # block edge in pixels for that split
RANDOM_SEED = 0             # same seed + same data = same model

# --------------------------------------------------------------------------- preprocessing
# These choices are saved with every model and repeated automatically at prediction time.
USE_DEM = "auto"            # "auto" = use terrain features when EVERY area has dem.tif; True / False to force
DEFAULT_SENSOR = "auto"     # pixel scaling when an area has no area.json: "auto", "reflectance",
                            # "sentinel2_composite", "sentinel2", "landsat89" (see rockmap/config.py SENSORS)
MASK_SURFACES = True        # never learn from snow, water, dense vegetation, shadow or cloud pixels

# --------------------------------------------------------------------------- after training
ACTIVATE_NEW_MODEL = True   # make the new version the one the application uses (old versions are kept)
USE_CACHE = True            # reuse preprocessed samples from PROCESSED_DIR when nothing changed
