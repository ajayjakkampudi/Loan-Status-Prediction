import os, sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from src.exception import CustomeException
from src.logger import logging
from src.utils import save_object,evaluate_models

@dataclass
class ModelTrainConfig:
    trained_model_path = os.path.join('artifacts','model.pkl')
    

class ModelTrainer:
    def __init__(self) -> None:
        self.model_trainer_config = ModelTrainConfig()
        
    def initiate_model_trainer(self, train_arr, val_arr):
        ''' This fuction performes the training and evaluation of data and returns best accuracy score'''
        logging.info("Model Trainer initiated")
        
        try:
            X_train, y_train, X_val, y_val = train_arr[:,:-1], train_arr[:,-1], val_arr[:,:-1], val_arr[:,-1]
            logging.info("Data was seperated to X and y train and val")
            
            models = {
                'logistic_regression' : LogisticRegression(),
                'knn' : KNeighborsClassifier(n_neighbors = 7),
                'decision_tree' : DecisionTreeClassifier(),
                'random_forest' : RandomForestClassifier()
            }
            
            # Training and evaluating models
            train_report, val_report = evaluate_models(X_train, y_train, X_val, y_val,models)
            print("Train_report:\n",train_report)
            
            # Sorting val report wrt accuracy to know the best model
            sorted_val_report = dict(sorted(val_report.items(), key= lambda x: x[1]['val_accuracy'], reverse= True))
            print("Val_report:\n",sorted_val_report)
            logging.info("Val report was sorted wrt accuracy")
            best_model_name = list(sorted_val_report.keys())[0]
            
            # Obtained Best model
            best_model = models[best_model_name]
            logging.info(f"Obtained best model : {best_model_name}")
            
            # Saving the best model
            save_object(save_path= self.model_trainer_config.trained_model_path, 
                        object= best_model
                        )
            logging.info("Best Model was saved")
            
            y_pred = best_model.predict(X_val)
            accuracy = accuracy_score(y_val, y_pred)
            
            logging.info("Model Trainer Completed")
            return accuracy
            
        except CustomeException as ce:
            raise CustomeException(ce)