#importing dependencies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')


#reading the data
df = pd.read_csv('/home/sumit/PriceLens/data/raw/cardekho_imputated.csv')

#defining independent and dependent features
X = df.drop(['car_name','brand', 'selling_price'], axis=1)
y = df['selling_price']



def num_column_transformer(df):
    le = LabelEncoder()
    le_cols = ['model']
    return le.fit_transform(df[le_cols])


def cat_column_transformer(df):

    df['model']=num_column_transformer(df)

    ohe = OneHotEncoder()
    scaler = StandardScaler()

    ohe_cols = ['seller_type', 'fuel_type', 'transmission_type']
    scaler_cols = [column for column in df.columns if df[column].dtype != 'object']     #numerical columns

    #creating Column Transformer with 3 types of transformers
    preprocessor = ColumnTransformer(
        [
            ("OneHotEncoder", ohe, ohe_cols),
            ("StandardScaler", scaler, scaler_cols)
        ], remainder="passthrough"
    )
    
    return preprocessor.fit_transform(df)

cat_column_transformer(X)