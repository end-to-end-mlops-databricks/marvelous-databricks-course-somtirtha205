import pandas as pd
import yaml
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class DataProcessor:
    def __init__(self, filepath, config):
        self.df = self.load_data(filepath)
        self.config = config
        self.X = None
        self.y = None
        self.preprocessor = None
        self.df1 = self.remove_id()

    def remove_id(self):
        # Remove Booking_ID from features
         return self.df.drop(self.config['id'], axis=1)
        
    def load_data(self, filepath):
            return pd.read_csv(filepath)

    def preprocess_data(self):
        target = self.config['target']
        
        # Separate features and target
        self.X = self.df1[self.config['num_features'] + self.config['cat_features']]
        self.y = self.df1[target]
        
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
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, self.config['num_features']),
                ('cat', categorical_transformer, self.config['cat_features'])
            ])