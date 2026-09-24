"""Classification models: CNN (deep learning) and classical ML benchmarks (RF, SVM)."""
from .classical import ClassicalClassifier
from .cnn import CNNClassifier, LithoCNN

__all__ = ["ClassicalClassifier", "CNNClassifier", "LithoCNN"]
