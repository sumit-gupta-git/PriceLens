import pandas as pd
import numpy as np
import logging
import joblib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Prediction:
    def __init__(self):
        self.independent = ['vehicle_age', 'km_driven', 'mileage', 'engine', 'max_power', 'seats',
                             'km_per_year', 'power_to_engine_ratio',
                            'efficiency_score', 'is_premium_brand', 'is_popular_brand',
                            'brand_encoded', 'seller_type_encoded', 'fuel_type_encoded',
                            'transmission_type_encoded', 'age_category_encoded',
                            'mileage_category_encoded', 'engine_category_encoded',
                            'power_category_encoded', 'fuel_efficiency_encoded']
        
        self.dependent = ['selling_price']
        self.model = {}
        self.X_test = {}
        self.y_test = {}
        self.X_train = {}
        self.y_train = {}

    
    def model_training(self, df):
        #importing dependencies
        from sklearn.model_selection import train_test_split, RandomizedSearchCV
        from sklearn.ensemble import RandomForestRegressor

        X = df[self.independent]
        y = df[self.dependent]

        #train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        #model_training with best parameters found in modeling.ipynb
        model = RandomForestRegressor(n_estimators=100, min_samples_split=2, 
                                    max_features=5, max_depth=None, 
                                    n_jobs=-1)
        model.fit(X_train, y_train)
        self.model = model

        logger.info('Model trained')

        return self.model
    
    
    ##Creating Function to Evaluate Model
    def evaluate_model(self, true, predicted):
        from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
        
        mae = mean_absolute_error(true, predicted)
        mse = mean_squared_error(true, predicted)
        rmse = np.sqrt(mse)
        r2_square = r2_score(true, predicted)
        logger.info('evaluation completed!')
        return mae, rmse, r2_square
    
if __name__ == '__main__':
    #loading dataset
    df = pd.read_csv('/home/sumit/PriceLens/data/processed/cardekho_encoded.csv')

    #model training
    p = Prediction()
    model = p.model_training(df)
    y_pred = model.predict(p.X_test)

    #evaluation
    mae, rmse, r2_score = p.evaluate_model(p.y_test, y_pred)
    print(mae, rmse, r2_score)
    print(type(model))
    joblib.dump(model, '/home/sumit/PriceLens/models/model.pkl')
    