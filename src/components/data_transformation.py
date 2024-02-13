import os
from typing import List, Tuple
import numpy as np
import warnings
warnings.filterwarnings('ignore')
from src.logger import logging
from src.exception import CustomeException
from src.utils import save_object, find_num_cat_col
import pandas as pd
import numpy as np
from src.config import Config
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass

@dataclass
class DataTransformConfig:
    preprocessor_obj_path = os.path.join('artifacts','preprocessor_transform.pkl')
    
class DataTransform:
    def __init__(self) -> None:
        self.transform_config = DataTransformConfig()
        
    def get_data_transform_obj(self, categorical_col: list, numerical_col: list):
        
        
        try:
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy= "median")),
                    ('scaler',StandardScaler())
                ]
            )
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy= "most_frequent")),
                    
                ]
            )
            
            preprocessor = ColumnTransformer([
                ('numerical_preprocessor', numerical_pipeline, numerical_col),
                ('categorical_preprocessor', categorical_pipeline, categorical_col)
            ]) 
            logging.info("Preprocessor was defined")
            
            return preprocessor
        
        except CustomeException as ce:
            raise CustomeException(ce)
        
    def initialize_data_transform(self, train_path, val_path) -> Tuple[np.array, np.array, str]:
        ''' This returns transformed train, val data and path where preprocessed object is saved '''
        try:
            train_df = pd.read_csv(train_path)
            val_df = pd.read_csv(val_path)
            logging.info("train and val data was read")
            
            input_feature_train_df = train_df.drop(columns=[Config.TARGET_FEATURE], axis = 1)
            target_feature_train_df = train_df[Config.TARGET_FEATURE]
            
            input_feature_val_df = val_df.drop(columns=[Config.TARGET_FEATURE], axis = 1)
            target_feature_val_df = val_df[Config.TARGET_FEATURE]
            logging.info("Dependent and independent variables are stored in varibles")
            
            # Obtains the columns which has categorical and numerical data in data frame 
            num_col, cat_col = find_num_cat_col(train_df)
            
            preprocessor_obj = self.get_data_transform_obj(categorical_col= cat_col, numerical_col= num_col)
            logging.info("Preproceesor object is initialized")
            
            input_train_arr = preprocessor_obj.fit_transform(input_feature_train_df)
            input_val_arr = preprocessor_obj.transform(input_feature_val_df)
            logging.info("Data is transformed")
            
            save_object(save_path= self.transform_config.preprocessor_obj_path, object=preprocessor_obj)
            logging.info("Preprocessor object is saved")
            
            train_arr = np.c_[input_train_arr, np.array(target_feature_train_df)]
            val_arr = np.c_[input_val_arr, np.array(target_feature_val_df)]
            
            logging.info("The tranformed independent features and dependent features are concatenated")
            return (
                train_arr,
                val_arr,
                self.transform_config.preprocessor_obj_path
            )
            
        except CustomeException as ce:
            raise CustomeException(ce)
        
        
            
            
            
            
     