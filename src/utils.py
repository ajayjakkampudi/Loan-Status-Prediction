import warnings
warnings.filterwarnings('ignore')
import pandas as pd
import os
import pickle as pkl
from src.config import Config
from src.logger import logging
from src.exception import CustomeException
from typing import Tuple, List

# Saves the object in pickle file
def save_object(save_path: str, object) -> None:
    ''' This fuction saves the object in pickle file '''
    try:
        dir_path = os.path.dirname(save_path)
        os.makedirs(dir_path, exist_ok= True)
        with open(save_path,'wb') as f:
            pkl.dump(object, f)
            
        logging.info(f'Object is save in {save_path}')
        
    except CustomeException as ce:
        raise CustomeException(ce)
    
# Gives the list of columns which has numeric and categorical data types 
def find_num_cat_col(df: pd.DataFrame) -> Tuple[List, List]:
    ''' On time being it lists the categories data which has only two unique values'''
    
    num_lst, cat_lst = [], []
    
    for col in df.columns:
        if col == Config.TARGET_FEATURE: continue
        if df[col].nunique() == 2: cat_lst.append(col)
        else: num_lst.append(col)
        
    print(num_lst)
    print(cat_lst)
    return num_lst, cat_lst
    