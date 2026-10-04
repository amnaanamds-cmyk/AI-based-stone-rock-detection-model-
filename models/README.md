# Trained models

Every training run (`python training/train.py`) saves a complete, self-contained model version. **Versions are never overwritten.**

```
models/trained/
├── model_v1/
│   ├── meta.json             everything needed to predict: classes, feature list, preprocessing
│   │                         (sensor scaling, masking, DEM yes/no, normaliser), metrics, training data
│   ├── cnn.pt                convolutional neural network (PyTorch weights)
│   ├── rf.joblib             Random Forest (scikit-learn)
│   ├── svm.joblib            Support Vector Machine (scikit-learn)
│   ├── evaluation.txt        readable evaluation report (accuracy, kappa, per class, confusion matrix)
│   ├── evaluation.json       the same numbers for programs
│   ├── cm_cnn.png …          confusion-matrix pictures
│   └── training_config.json  settings used for this run (epochs, learning rate, split, seed, data)
├── model_v2/
└── current.json              {"version": "model_v2"}  <- the version the application uses
```

## Which model does the application use?

The one named in `current.json`. After training, the new version becomes current automatically, unless you use `--no-activate` or set `ACTIVATE_NEW_MODEL = False`. The web application checks this file every few seconds and the prediction module reads it on every start, so **you never edit application code after training**. Inside each version, the application uses the model type chosen at training: the best kappa, or `APP_MODEL_TYPE` in `training/config.py`. This is stored as `app_algorithm` in `meta.json`.

```bash
python training/models.py list              # all versions, * = current
python training/models.py info model_v2     # evaluation report of a version
python training/models.py use model_v1      # switch back (rollback)
python training/models.py delete model_v1   # remove an old version (never the current one)
```

## Notes
- Model files are **not committed to git**, because they are large and depend on your data. Back up the versions you want to keep, or copy a version folder to another computer: the whole folder is all that is needed.
- The folder can be moved with the environment variable `ROCKMAP_MODELS_DIR`.
- Models trained in the web dashboard (scene or region training) live in the dashboard's data folder. Models from this folder appear in the dashboard's model list as "training module".
