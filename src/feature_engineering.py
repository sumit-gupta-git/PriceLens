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
        else:
            # If brand column doesn't exist, create default features
            logger.warning("Brand column not found, creating default brand features")
            df_features['is_premium_brand'] = 0
            df_features['is_popular_brand'] = 0
        
        return df_features

    def encode_categorical_features(self, data):
        processed_data = data.copy()
        
        # Original categorical features encoding
        if 'brand' in processed_data.columns:
            brand_mapping = {
                'Maruti': 0, 'Hyundai': 1, 'Honda': 2, 'Toyota': 3, 'Ford': 4,
                'Mahindra': 5, 'Renault': 6, 'Nissan': 7, 'Tata': 8, 'Volkswagen': 9,
                'BMW': 10, 'Mercedes': 11, 'Audi': 12, 'Other': 13
            }
            processed_data['brand_encoded'] = processed_data['brand'].map(brand_mapping).fillna(13)
        else:
            processed_data['brand_encoded'] = 13  # Default to 'Other'
        
        if 'seller_type' in processed_data.columns:
            seller_mapping = {'Dealer': 0, 'Individual': 1, 'TrustMark': 2}
            processed_data['seller_type_encoded'] = processed_data['seller_type'].map(seller_mapping).fillna(1)
        else:
            processed_data['seller_type_encoded'] = 1  # Default
        
        if 'fuel_type' in processed_data.columns:
            fuel_mapping = {'Petrol': 0, 'Diesel': 1, 'CNG': 2, 'LPG': 3, 'Electric': 4}
            processed_data['fuel_type_encoded'] = processed_data['fuel_type'].map(fuel_mapping).fillna(0)
        else:
            processed_data['fuel_type_encoded'] = 0  # Default
        
        if 'transmission_type' in processed_data.columns:
            transmission_mapping = {'Manual': 0, 'Automatic': 1}
            processed_data['transmission_type_encoded'] = processed_data['transmission_type'].map(transmission_mapping).fillna(0)
        else:
            processed_data['transmission_type_encoded'] = 0  # Default
        
        # NEW: Encode categorical features created by feature engineering functions
        
        # Age category encoding (from create_age_features)
        if 'age_category' in processed_data.columns:
            age_mapping = {'New': 0, 'Moderate': 1, 'Old': 2, 'Very_Old': 3}
            processed_data['age_category_encoded'] = processed_data['age_category'].map(age_mapping).fillna(2)
        else:
            processed_data['age_category_encoded'] = 2  # Default to 'Old'
        
        # Mileage category encoding (from create_mileage_features)
        if 'mileage_category' in processed_data.columns:
            mileage_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
            processed_data['mileage_category_encoded'] = processed_data['mileage_category'].map(mileage_mapping).fillna(1)
        else:
            processed_data['mileage_category_encoded'] = 1  # Default to 'Medium'
        
        # Engine category encoding (from create_engine_features)
        if 'engine_category' in processed_data.columns:
            engine_mapping = {'Small': 0, 'Medium': 1, 'Large': 2, 'Very_Large': 3}
            processed_data['engine_category_encoded'] = processed_data['engine_category'].map(engine_mapping).fillna(1)
        else:
            processed_data['engine_category_encoded'] = 1  # Default to 'Medium'
        
        # Power category encoding (from create_engine_features)
        if 'power_category' in processed_data.columns:
            power_mapping = {'Low': 0, 'Medium': 1, 'High': 2, 'Very_High': 3}
            processed_data['power_category_encoded'] = processed_data['power_category'].map(power_mapping).fillna(1)
        else:
            processed_data['power_category_encoded'] = 1  # Default to 'Medium'
        
        # Fuel efficiency category encoding (from create_efficiency_features)
        if 'fuel_efficiency' in processed_data.columns:
            efficiency_mapping = {'Poor': 0, 'Average': 1, 'Good': 2, 'Excellent': 3}
            processed_data['fuel_efficiency_encoded'] = processed_data['fuel_efficiency'].map(efficiency_mapping).fillna(1)
        else:
            processed_data['fuel_efficiency_encoded'] = 1  # Default to 'Average'
        
        # Drop all categorical columns (both original and created)
        categorical_columns_to_drop = [
            'brand', 'fuel_type', 'seller_type', 'transmission_type',  # Original
            'age_category', 'mileage_category', 'engine_category', 
            'power_category', 'fuel_efficiency'  # Created by feature engineering
        ]
        
        columns_to_drop = []
        for col in categorical_columns_to_drop:
            if col in processed_data.columns:
                columns_to_drop.append(col)
        
        if columns_to_drop:
            processed_data = processed_data.drop(columns=columns_to_drop, axis=1)
            logger.info(f'Dropped categorical columns: {columns_to_drop}')
        
        logger.info("All categorical features encoded successfully")
        return processed_data

    
    def apply(self, df):
        logger.info("Starting feature engineering...")
        
        df_engineered = df.copy()
        
        df_engineered = self.create_age_features(df_engineered)
        df_engineered = self.create_mileage_features(df_engineered)
        df_engineered = self.create_engine_features(df_engineered)
        df_engineered = self.create_efficiency_features(df_engineered)
        df_engineered = self.create_brand_features(df_engineered)
        df_engineered = self.encode_categorical_features(df_engineered)
        
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
        print(df.columns)

        #exporting the data
        df.to_csv('/home/sumit/PriceLens/data/processed/cardekho_encoded.csv')
        
    except Exception as e:
        print(f'Error : {e}')