#importing dependencies
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def validate_data():
    #reading the data
    df = pd.read_csv('/home/sumit/PriceLens/data/raw/cardekho_imputated.csv')

    #dropping unnecessary columns
    df.drop(['Unnamed: 0','car_name'], axis=1, inplace=True)

    return df


def analyze_categorical_features(df):
    cat_features = [feature for feature in df.columns if df[feature].dtype == "object"]
    print(f"Number of categorical features are {len(cat_features)} and name of those features are {cat_features}")
    d = {}
    for feature in cat_features:
        d[feature] = dict(df[feature].value_counts())
    return d

def analyze_numerical_features(df):
    num_features = [feature for feature in df.columns if df[feature].dtype != "object"]
    print(f"Number of numerical features are {len(num_features)} and name of those features are {num_features}")
    
    return df[num_features].corr()

df = validate_data()
print(analyze_categorical_features(df))
print()
print(analyze_numerical_features(df))
