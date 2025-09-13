
__version__ = "1.0.0"
__author__ = "Car Price Predictor Team"

# Import main classes for easy access
from .prediction_service import Prediction
from .data_preprocessing import DataPreprocessor
from .feature_engineering import FeatureEngineer

__all__ = [
    'Prediction', 
    'DataPreprocessor',
    'FeatureEngineer'
]