import warnings
warnings.filterwarnings('ignore') 
from src.exception import CustomeException
import pandas as pd
from src.utils import load_object
from src.config import Config

config = Config()
class PredictPipeline:
    def __init__(self) -> None:
        self.model_path = config.model_path
        self.preprocessor_path = config.preprocessor_path
    
    def predict(self, features):
        try:
            model = load_object(self.model_path)
            preprocessor = load_object(self.preprocessor_path)
            data_transformed = preprocessor.transform(features)
            y_prediction = model.predict(data_transformed)
            return y_prediction
        
        except CustomeException as ce:
            raise ce
            
class CustomData:
    def __init__(self,
                 battery_power: int, bluetooth: int, clock_speed: float, dual_sim: int, front_camera_pixels: int,
                 four_g: int, int_memory: int, m_dep: float, mobile_wt: int, n_cores: int,primary_camera_pixels: int,
                 px_height: int, px_width: int,ram: int, screen_height: int, screen_width: int, talk_time: int,
                 three_g: int,touch_screen: int, wifi: int):
        try: 
            self.custom_data_input_dict = {
                'battery_power': [battery_power],
                'blue': [bluetooth],
                'clock_speed': [clock_speed],
                'dual_sim': [dual_sim],
                'fc': [front_camera_pixels],
                'four_g': [four_g],
                'int_memory': [int_memory],
                'm_dep': [m_dep],
                'mobile_wt': [mobile_wt],
                'n_cores': [n_cores],
                'pc': [primary_camera_pixels],
                'px_height': [px_height],
                'px_width': [px_width],
                'ram': [ram],
                'sc_h': [screen_height],
                'sc_w': [screen_width],
                'talk_time': [talk_time],
                'three_g': [three_g],
                'touch_screen': [touch_screen],
                'wifi': [wifi],   
            }
            
        except CustomeException as ce:
            raise ce
        
        
    def get_features_as_df(self):
        
        try:
            return pd.DataFrame(self.custom_data_input_dict)
        
        except CustomeException as ce:
            raise ce
