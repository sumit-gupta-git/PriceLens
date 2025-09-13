import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataPreprocessor:
    
    def __init__(self):
        self.required_columns = [
            'brand', 'vehicle_age', 'km_driven', 'seller_type',
            'fuel_type', 'transmission_type', 'mileage', 'engine',
            'max_power', 'seats', 'selling_price'
        ]
        self.categorical_columns = [
            'brand', 'model', 'seller_type', 'fuel_type', 'transmission_type'
        ]
        self.numerical_columns = [
            'vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats'
        ]
    
    def load_data(self, file_path: str):
        try:
            df = pd.read_csv(file_path)
            logger.info(f"Data loaded successfully. Shape: {df.shape}")
            
            # Remove unnamed index columns if present
            unnamed_cols = [col for col in df.columns if col.startswith('Unnamed')]
            if unnamed_cols:
                df.drop(columns=unnamed_cols, inplace=True)
                logger.info(f"Removed unnamed columns: {unnamed_cols}")
            
            return df
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df):
        missing_cols = set(self.required_columns) - set(df.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        logger.info("Data validation passed")
        return True
    
    def clean_data(self, df):
        df_cleaned = df.copy()
        
        # Handle missing values
        logger.info("Handling missing values...")
        
        for col in self.numerical_columns:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                median_val = df_cleaned[col].median()
                df_cleaned[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing values in {col} with median: {median_val}")
        
        for col in self.categorical_columns:
            if col in df_cleaned.columns and df_cleaned[col].isnull().any():
                mode_val = df_cleaned[col].mode()[0] if not df_cleaned[col].mode().empty else 'Unknown'
                df_cleaned[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing values in {col} with mode: {mode_val}")
        
        logger.info("Removing extreme outliers...")
        initial_rows = len(df_cleaned)
        
        for col in ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power']:
            if col in df_cleaned.columns:
                mean_val = df_cleaned[col].mean()
                std_val = df_cleaned[col].std()
                lower_bound = mean_val - 3 * std_val
                upper_bound = mean_val + 3 * std_val
                
                outliers_mask = (df_cleaned[col] < lower_bound) | (df_cleaned[col] > upper_bound)
                df_cleaned = df_cleaned[~outliers_mask]
        
        final_rows = len(df_cleaned)
        logger.info(f"Removed {initial_rows - final_rows} outlier rows")
        
        return df_cleaned
    
    def preprocess_features(self, df):
        df_processed = df.copy()
        
        # Convert string columns that should be numeric
        numeric_string_cols = ['mileage', 'engine', 'max_power']
        for col in numeric_string_cols:
            if col in df_processed.columns:
                # Remove units and convert to float
                if df_processed[col].dtype == 'object':
                    df_processed[col] = pd.to_numeric(
                        df_processed[col].astype(str).str.extract(r'(\d+\.?\d*)')[0], 
                        errors='coerce'
                    )
        
        for col in self.numerical_columns:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
        
        for col in self.categorical_columns:
            if col in df_processed.columns:
                df_processed[col] = df_processed[col].astype(str)
        
        logger.info("Feature preprocessing completed")
        return df_processed
    
    def get_data_summary(self, df):
        summary = {
            'shape': df.shape,
            'missing_values': df.isnull().sum().to_dict(),
            'data_types': df.dtypes.to_dict(),
            'numerical_summary': df.describe().to_dict(),
            'categorical_summary': {}
        }

        for col in self.categorical_columns:
            if col in df.columns:
                summary['categorical_summary'][col] = df[col].value_counts().head().to_dict()
        
        return summary


class CustomCategoricalTransformer(BaseEstimator, TransformerMixin):

    def __init__(self, method='label_encode'):
        self.method = method
        self.encoders = {}
        self.fitted = False
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import LabelEncoder, OneHotEncoder
        
        if isinstance(X, pd.DataFrame):
            for col in X.columns:
                if X[col].dtype == 'object' or X[col].dtype.name == 'category':
                    if self.method == 'label_encode':
                        encoder = LabelEncoder()
                        encoder.fit(X[col].astype(str))
                        self.encoders[col] = encoder
                        logger.info(f"Fitted LabelEncoder for column: {col}")
                    
                    elif self.method == 'onehot':
                        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
                        encoder.fit(X[[col]])
                        self.encoders[col] = encoder
                        logger.info(f"Fitted OneHotEncoder for column: {col}")
            
            self.fitted = True
        
        return self  
    
    def transform(self, X):
        
        if not self.fitted:
            raise ValueError("Transformer must be fitted before calling transform")
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input must be a pandas DataFrame")
        
        X_transformed = X.copy()
        
        for col in X_transformed.columns:
            if col in self.encoders:
                try:
                    X_transformed[col] = self.encoders[col].transform(X_transformed[col].astype(str))
                    logger.info(f"Applied LabelEncoder to column: {col}")
                except ValueError as e:
                    logger.warning(f"Unseen categories in {col}: {e}")
                    unseen_mask = ~X_transformed[col].astype(str).isin(self.encoders[col].classes_)
                    if unseen_mask.any():
                        most_frequent_class = self.encoders[col].classes_[0]
                        X_transformed.loc[unseen_mask, col] = most_frequent_class
                        logger.info(f"Replaced {unseen_mask.sum()} unseen values in {col}")
                    X_transformed[col] = self.encoders[col].transform(X_transformed[col].astype(str))
        
        logger.info("Categorical transformation completed")
        return X_transformed  

    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


class CustomNumericalTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, method='standard'):
        self.method = method
        self.scaler = None
        self.fitted = False
    
    def fit(self, X, y=None):
        from sklearn.preprocessing import StandardScaler, MinMaxScaler
        
        if self.method == 'standard':
            self.scaler = StandardScaler()
        elif self.method == 'minmax':
            self.scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        if isinstance(X, pd.DataFrame):
            self.scaler.fit(X)
        else:
            self.scaler.fit(X)
        
        self.fitted = True
        logger.info(f"Fitted {self.method} scaler for numerical data")
        return self
    
    def transform(self, X):
        if not self.fitted:
            raise ValueError("Transformer must be fitted before calling transform")
        
        if isinstance(X, pd.DataFrame):
            scaled_data = self.scaler.transform(X)
            return pd.DataFrame(scaled_data, columns=X.columns, index=X.index)
        else:
            return self.scaler.transform(X)
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)


def cat_column_transformer(X):
    transformer = CustomCategoricalTransformer(method='label_encode')
    return transformer.fit_transform(X)


def num_column_tranformer(X):
    transformer = CustomNumericalTransformer(method='standard')
    return transformer.fit_transform(X)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    try:
        df = preprocessor.load_data('/home/sumit/PriceLens/data/processed/cardekho_imputated.csv')
        preprocessor.validate_data(df)
        cleaned_data = preprocessor.clean_data(df)
        processed_data = preprocessor.preprocess_features(cleaned_data)
        summary = preprocessor.get_data_summary(processed_data)
        
        logger.info('Preprocessing done!')
        print(f"Final data shape: {summary['shape']}")

        processed_data.to_csv('/home/sumit/PriceLens/data/processed/cardekho_imputated.csv', )
        
    except Exception as e:
        print(f" DataPreprocessor test failed: {str(e)}")
