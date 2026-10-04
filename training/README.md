# Training module

Training is completely separate from the application. The application only **loads** the model that this module produces, and it never trains.

```
dataset/raw/  ──►  training/train.py  ──►  models/trained/model_vN/  ──►  application / prediction/predict.py
 (your data)     preprocess · train ·       (versioned, current.json        load · preprocess (same) ·
                 evaluate · save            says which one is used)         predict · show
```

## Quick start

```bash
# 0. once: install (run.sh / run.bat do this), or: pip install -e .
# 1. put your data in dataset/raw/<area_name>/   (format: dataset/README.md)
python training/train.py --check     # 2. check the dataset
python training/train.py             # 3. preprocess, train, evaluate, save, activate
python training/models.py list       # 4. see all versions (* = used by the application)
python prediction/predict.py dataset/raw/<area>/image.tif --dem dataset/raw/<area>/dem.tif   # 5. try it
```

No real data yet? `python training/make_sample_dataset.py` creates a **synthetic** sample area so you can see the whole pipeline work. A model trained on it is flagged as sample data; it is not a real geological model.

Common options (anything else is set in `config.py`):

| Command | Effect |
|---|---|
| `python training/train.py --data path/to/folder` | train on another dataset folder |
| `python training/train.py --epochs 50 --lr 0.001` | change CNN training for this run |
| `python training/train.py --models cnn rf` | train only some model types (faster) |
| `python training/train.py --samples 8000` | more training pixels per class |
| `python training/train.py --no-activate` | save the new version without making it current |
| `python training/train.py --no-cache` | redo preprocessing for every area |
| `python training/evaluate.py --data path/to/area` | test the current model on an independent labelled area |
| `python training/models.py use model_v1` | switch the application to another version |

## Files

| File | What it does | Edit it? |
|---|---|---|
| `config.py` | **all settings**: paths, model types, epochs, batch size, learning rate, samples, split, seed, DEM, sensor | yes, this is the place |
| `train.py` | the six steps: find data → check → preprocess → train → evaluate → save and activate | rarely |
| `dataset.py` | finds areas in `dataset/raw`, checks them, understands `area.json` | to support new file names or formats |
| `preprocessing.py` | one area → samples (reflectance, masks, features, block split), with a cache in `dataset/processed` | rarely |
| `evaluate.py` | metrics, `evaluation.txt`, testing on an independent area | to add metrics |
| `models.py` | list / info / use / delete model versions | no |
| `make_sample_dataset.py` | synthetic sample data | no |

The actual algorithms live in the core package and are shared with the application, so training and prediction always do the same thing:
- `rockmap/models/cnn.py`: the CNN (PyTorch)
- `rockmap/models/classical.py`: Random Forest and SVM
- `rockmap/features.py`: features: 6 bands, 7 spectral indices, 5 terrain features
- `rockmap/preprocessing.py`: reflectance scaling, cloud/snow/water/vegetation/shadow masks
- `rockmap/pipeline.py`: sampling (`SampleCollector`), fitting (`fit_bundle`) and prediction (`ModelBundle`)
- `rockmap/model_store.py`: model versions and `current.json`

## How training works

1. **Preprocessing.** Each image is converted to surface reflectance (the sensor preset is detected or set in `area.json`). Cloud, snow, water, dense vegetation and deep shadow are masked. 13 features per pixel are computed, or 18 with a DEM.
2. **Samples.** The labelled pixels are split into **train / validation / test by square blocks** (`BLOCK_SIZE`, `SPLIT`). Test pixels therefore come from places the model never saw, which gives honest accuracy for maps. Up to `SAMPLES_PER_CLASS` training pixels are drawn per class. Pixels on unit boundaries (where maps are least reliable) are skipped for training.
3. **Normalisation.** Each feature is scaled to mean 0, standard deviation 1 using the training pixels. The means and standard deviations are saved in `meta.json` and reused at prediction time.
4. **Models.**
   - **CNN:** 9×9-pixel patches go through a 1×1 "spectral" layer, four 3×3 convolutions with batch normalisation, dropout and a 1×1 classifier head. Training uses class-weighted cross-entropy, AdamW with a one-cycle learning rate, and random flips and rotations. The epoch with the best validation accuracy is kept.
   - **Random Forest** (200 trees) and **SVM** (RBF kernel): trained on the same pixels as benchmarks.
5. **Evaluation** on the test blocks: overall accuracy, Cohen's kappa, precision, recall and F1 per class, and the confusion matrix. These are written to `evaluation.txt` with warnings, e.g. sample data, kappa below 0.6, or weak classes.
6. **Save and activate:** the model goes to `models/trained/model_vN/`, and `current.json` is updated.

**Reading the results.** Kappa above 0.6 is usable and above 0.8 is very good. Look at the per-class F1 and the confusion matrix to see which rocks are mixed up. These numbers are measured against your labels, so if the labels are wrong or generalised, the numbers inherit that. The most honest test is `python training/evaluate.py --data <an area not used for training>`.

## Retraining: from scratch, on all data

Every run trains **from scratch on all areas in `dataset/raw/`**: the old data plus the new. This is the right choice here because:
- the CNN is small and trains in minutes on a normal CPU (a GPU is used automatically if present)
- Random Forest and SVM cannot be "continued"; they are always refitted
- training on everything avoids the model forgetting old areas, which fine-tuning on new data alone can cause
- the normaliser and class list are rebuilt from all the data

```
first dataset ─► train ─► model_v1 (current)
add areas     ─► train ─► model_v2 (current; v1 kept)
v2 worse?     ─► python training/models.py use model_v1
```

Unchanged areas are not reprocessed: their samples come from `dataset/processed/`. Delete that folder at any time; it is rebuilt automatically.

## Changing settings

Open `training/config.py`. Every setting has a comment. Typical changes:
- **more epochs** (`EPOCHS`) if the validation accuracy is still rising at the end
- **more samples** (`SAMPLES_PER_CLASS`) when you have lots of labels
- **a bigger CNN** (`CNN_WIDTH`)
- **which model the app uses** (`APP_MODEL_TYPE`)
- the **split** and **block size**

Run `python training/train.py` again afterwards. The settings of every run are saved in `training_config.json` inside the model folder.

## Adding a new class

1. Add one line to `ROCK_CLASSES` in **`rockmap/config.py`**, with a new id (8, 9, …; below 250), a name, a map colour and a description:
   ```python
   RockClass(8, "Quartzite", "#c9b8a6", "Recrystallised quartz sandstone, very bright, iron-poor."),
   ```
2. Label it in your data with that id (or map units to it in `area.json`).
3. Run `python training/train.py`.

Nothing else is needed. The network's output layer is sized automatically from the classes found in the training data, and maps, legends, statistics and reports read the class list from `rockmap/config.py`. Old model versions keep working with the classes they were trained on. Optionally, give the new class a rock-strength value in `LITHO_WEAKNESS` (`rockmap/analytics.py`) so the landslide map uses it. Unknown classes are simply skipped there.

## Changing the model architecture

| Change | File |
|---|---|
| CNN layers, filters, dropout | `rockmap/models/cnn.py` (`LithoCNN`). If you change the receptive field (the number of 3×3 layers), also change `DEFAULT_PATCH_SIZE` in `rockmap/config.py` and `PATCH_SIZE` in `training/config.py` |
| CNN optimiser, loss, augmentation | `rockmap/models/cnn.py` (`CNNClassifier.fit`, `_augment`) |
| Random Forest / SVM hyper-parameters | `rockmap/models/classical.py` |
| Input features (new indices) | `rockmap/features.py` (`spectral_indices`, `terrain_features`). Old model versions keep their own feature list in `meta.json`, but they need the features they were trained with |
| A completely new model type | add a class with `fit` / `predict_proba` / `save` / `load` next to the others, then register it in `ALGORITHMS` in `rockmap/config.py` and in `fit_bundle` / `ModelBundle.model` in `rockmap/pipeline.py` |

After any change, run `python -m pytest` and train again.
