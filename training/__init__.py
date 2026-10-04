"""RockMap training module: dataset -> preprocessing -> training -> evaluation -> versioned model.

Run ``python training/train.py`` (see training/README.md). The web application never trains; it
loads the current model from ``models/trained/`` (see ``rockmap/model_store.py``).
"""
