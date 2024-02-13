import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import os
import shutil
from src.exception import CustomeException
from src.logger import logging
from sklearn.model_selection import train_test_split
from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    ''' Store the path of train, val, test'''
    raw_path = os.path.join('artifacts','train.csv')
    train_path = os.path.join('artifacts','train.csv')
    val_path = os.path.join('artifacts','val.csv')
    test_path = os.path.join('artifacts','test.csv')
    
class DataIngestion:
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig()
        
    def data_ingestion_initiate(self):
        logging.info("Data Ingestion Initiated")
        
        try:
            df = pd.read_csv('notebook/data/train.csv')
            logging.info("CSV was read")
            
            # Storing raw data in artifacts
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_path)), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_path,header=True,index=False)
            logging.info('Raw data is save in artifacts')
            
            # Splitting data to traiin and val
            train_df, val_df = train_test_split(df,test_size=0.2,random_state=100)
            logging.info('Data was seperated to train and val')
            
            # Storing train, val and test in artifacts
            ## train
            train_df.to_csv(self.ingestion_config.train_path,header=True,index=False)
            logging.info('train data is save in artifacts')
            
            ## Val
            val_df.to_csv(self.ingestion_config.val_path,header=True,index=False)
            logging.info('val data is save in artifacts')
            
            ## Test
            shutil.copy('notebook/data/train.csv',self.ingestion_config.test_path)
            logging.info('test data is saved in artifacts')
            
            logging.info("Data Ingestion is completed")
            
            return (
                self.ingestion_config.train_path,
                self.ingestion_config.val_path,
                self.ingestion_config.test_path
            )
            
        except CustomeException as ce:
            raise CustomeException(ce)


    