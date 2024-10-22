# Databricks notebook source
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# COMMAND ----------

def load_data(filepath):
    """
    Load the data from the given filepath.
    """
    df = pd.read_csv(filepath)
    return df

filepath = "/Volumes/mlops_dev/hotel_reservation/data/Data.csv"
df = load_data(filepath)

# COMMAND ----------

def preprocess_data(df, target_column='booking_status', column='Booking_ID'):
    """
    Preprocess the data.
    
    Args:
    df (pandas.DataFrame): The input dataframe
    target_column (str): The name of the target column (default is 'booking_status')
    
    Returns:
    X (pandas.DataFrame): The feature dataframe
    y (pandas.Series): The target series
    preprocessor (ColumnTransformer): The preprocessing pipeline
    """
    
    # Remove Booking_ID from features
    df1 = df.drop(column, axis=1)
    
    # Separate features and target
    X = df1.drop(target_column, axis=1)
    y = df1[target_column]

    # Convert target to Numeric and change the column name back to booking_status
    y = pd.get_dummies(y, columns=target_column, drop_first=True)
    y.set_axis(["booking_status"], axis="columns", inplace=True)
    
    # Identify numeric and categorical columns
    numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
    categorical_features = X.select_dtypes(include=['object']).columns
    
    # Create preprocessing steps for numeric and categorical data
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])
    
    print("Features shape:", X.shape)
    print("Target shape:", y.shape)
    
    return X, y, preprocessor

X, y, preprocessor = preprocess_data(df)
