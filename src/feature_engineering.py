import pandas as pd
import numpy as np
import logging
from sklearn.base import BaseEstimator, TransformerMixin
from src.data_preprocessing import DataPreprocessor, CustomCategoricalTransformer, CustomNumericalTransformer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        self.fitted = False
    
    def create_age_features(self, df):
        df_features = df.copy()
        
        if 'vehicle_age' in df_features.columns:
            # Age categories
            df_features['age_category'] = pd.cut(
                df_features['vehicle_age'],
                bins=[0, 3, 7, 12, float('inf')],
                labels=['New', 'Moderate', 'Old', 'Very_Old']
            )
            
            logger.info("Age features created")
        
        return df_features
    
    def create_mileage_features(self, df):
        df_features = df.copy()
        
        if 'km_driven' in df_features.columns and 'vehicle_age' in df_features.columns:
            # Average kilometers per year
            df_features['km_per_year'] = df_features['km_driven'] / (df_features['vehicle_age'] + 1)
            
            # Mileage categories
            df_features['mileage_category'] = pd.cut(
                df_features['km_driven'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
            logger.info("Mileage features created")
        
        return df_features
    
    def create_engine_features(self, df):
        df_features = df.copy()
        
        if 'engine' in df_features.columns and 'max_power' in df_features.columns:
            # Power to engine ratio
            df_features['power_to_engine_ratio'] = df_features['max_power'] / (df_features['engine'] + 1)
            
            # Engine categories
            df_features['engine_category'] = pd.cut(
                df_features['engine'],
                bins=[0, 1000, 1500, 2000, float('inf')],
                labels=['Small', 'Medium', 'Large', 'Very_Large']
            )
            
            # Power categories
            df_features['power_category'] = pd.cut(
                df_features['max_power'],
                bins=[0, 75, 125, 200, float('inf')],
                labels=['Low', 'Medium', 'High', 'Very_High']
            )
            
            logger.info("Engine features created")
        
        return df_features
    
    def create_efficiency_features(self, df):
        df_features = df.copy()
        
        if 'mileage' in df_features.columns and 'engine' in df_features.columns:
            # Efficiency score (mileage per engine size)
            df_features['efficiency_score'] = df_features['mileage'] / (df_features['engine'] / 1000)
            
            # Fuel efficiency categories
            df_features['fuel_efficiency'] = pd.cut(
                df_features['mileage'],
                bins=[0, 15, 20, 25, float('inf')],
                labels=['Poor', 'Average', 'Good', 'Excellent']
            )
            
            logger.info("Efficiency features created")
        
        return df_features
    
    def create_brand_features(self, df):
        df_features = df.copy()
        
        if 'brand' in df_features.columns:
            # Premium brand indicator
            premium_brands = ['BMW', 'Mercedes-Benz', 'Audi', 'Jaguar', 'Land Rover', 'Volvo']
            df_features['is_premium_brand'] = df_features['brand'].isin(premium_brands).astype(int)
            
            # Popular brand indicator
            popular_brands = ['Maruti', 'Hyundai', 'Honda', 'Toyota', 'Ford']
            df_features['is_popular_brand'] = df_features['brand'].isin(popular_brands).astype(int)
            
            logger.info("Brand features created")
        
        return df_features
    
    def scale_features(self, df, dtype, fit=True):
        df_feature = df.copy()
        
        if dtype == 'object':
           
            if fit:
                transformer = CustomCategoricalTransformer(method='label_encode')
                encoded_features = transformer.fit_transform(df_feature)
                self.encoders['categorical'] = transformer
                logger.info("Fitted categorical encoder")
                return encoded_features
            else:
                if 'categorical' not in self.encoders:
                    raise ValueError("Categorical encoder not fitted yet")
                encoded_features = self.encoders['categorical'].transform(df_feature)
                logger.info("Applied fitted categorical encoder")
                return encoded_features
        
        else:
            if fit:
                transformer = CustomNumericalTransformer(method='standard')
                scaled_features = transformer.fit_transform(df_feature)
                self.scalers['numerical'] = transformer
                logger.info("Fitted numerical scaler")
                return scaled_features
            else:
                if 'numerical' not in self.scalers:
                    raise ValueError("Numerical scaler not fitted yet")
                scaled_features = self.scalers['numerical'].transform(df_feature)
                logger.info("Applied fitted numerical scaler")
                return scaled_features
    
    def apply(self, df):
        logger.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        df_engineered = self.create_age_features(df_engineered)
        df_engineered = self.create_mileage_features(df_engineered)
        df_engineered = self.create_engine_features(df_engineered)
        df_engineered = self.create_efficiency_features(df_engineered)
        df_engineered = self.create_brand_features(df_engineered)
        
        self.feature_names = df_engineered.columns.tolist()
        
        logger.info(f"Feature engineering completed. Total features: {len(self.feature_names)}")
        return df_engineered
    
    def get_feature_summary(self, df: pd.DataFrame) :
        summary = {
            'total_features': len(df.columns),
            'numerical_features': len(df.select_dtypes(include=[np.number]).columns),
            'categorical_features': len(df.select_dtypes(include=['object', 'category']).columns),
            'feature_names': df.columns.tolist(),
            'feature_types': df.dtypes.to_dict()
        }
        
        return summary


class AdvancedFeatureTransformer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_engineer = FeatureEngineer()
        self.fitted = False
        self.numerical_cols = None
        self.categorical_cols = None
    
    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            X_engineered = self.feature_engineer.apply(X)
            
            self.numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
            self.categorical_cols = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
            
            logger.info(f"Found {len(self.numerical_cols)} numerical and {len(self.categorical_cols)} categorical columns")
            
            if len(self.numerical_cols) > 0:
                self.feature_engineer.scale_features(
                    X_engineered[self.numerical_cols], 
                    dtype='numerical', 
                    fit=True  
                )
                logger.info("Fitted numerical scaler")
            
            if len(self.categorical_cols) > 0:
                self.feature_engineer.scale_features(
                    X_engineered[self.categorical_cols], 
                    dtype='object', 
                    fit=True 
                )
                logger.info(" Fitted categorical encoder")
            
            self.fitted = True
            print(f"Transformer fitted successfully on {X.shape[0]} samples")
        
        return self
    
    def transform(self, X):
    
        if not self.fitted:
            raise ValueError("Transformer must be fitted before calling transform()")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        # Apply feature engineering
        X_engineered = self.feature_engineer.apply(X)
        
        # Transform features using fitted transformers
        current_numerical_cols = X_engineered.select_dtypes(include=[np.number]).columns.tolist()
        current_categorical_cols = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
        
        if len(current_numerical_cols) > 0:
            scaled_features = self.feature_engineer.scale_features(
                X_engineered[current_numerical_cols],
                dtype='numerical',
                fit=False
            )
            X_engineered[current_numerical_cols] = scaled_features

        if len(self.categorical_cols) > 0:
            categorical_cols_present = [col for col in self.categorical_cols if col in X_engineered.columns]
            if categorical_cols_present:
                logger.info(f"Encoding {len(categorical_cols_present)} categorical columns: {categorical_cols_present}")
                
                encoded_features = self.feature_engineer.scale_features(
                    X_engineered[categorical_cols_present],
                    dtype='object',
                    fit=False
                )
                
                X_engineered[categorical_cols_present] = encoded_features
                logger.info(f"Successfully encoded categorical features")

        return X_engineered
    
    def fit_transform(self, X, y=None):
        self.fit(X, y)
        return self.transform(X)
    
    def get_feature_names_out(self, input_features=None):
        if not self.fitted:
            raise ValueError("Transformer must be fitted before getting feature names")
        return self.feature_engineer.feature_names


if __name__ == "__main__":

    feature_engineer = FeatureEngineer()
    adv = AdvancedFeatureTransformer()
    
    try:
        #loading data
        preprocesser = DataPreprocessor()
        df = preprocesser.load_data('/home/sumit/PriceLens/data/processed/cardekho_imputated.csv')
        preprocesser.validate_data(df)

        #extracting features
        df = feature_engineer.apply(df)

        #scaling data
        df = adv.fit_transform(df)
        print(df.head())

        #exporting the data
        df.to_csv('/home/sumit/PriceLens/data/processed/cardekho_encoded.csv')
        
    except Exception as e:
        print(f'Error : {e}')