import warnings 
warnings.filterwarnings('ignore')
from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import shutil
from src.pipeline.predict_pipeline import CustomData, PredictPipeline
app = Flask(__name__)

# shutil.rmtree('logs')
# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods= ['GET',"POST"])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        data = CustomData(battery_power= request.form.get('battery_power'), bluetooth= request.form.get('bluetooth'), 
                          clock_speed= float(request.form.get('clock_speed')), dual_sim= request.form.get('dual_sim'),
                          front_camera_pixels= request.form.get('front_camera_pixels'), four_g= request.form.get('4G'),
                          int_memory= request.form.get('int_memory'), m_dep= float(request.form.get('m_dep')),
                          mobile_wt= request.form.get('mobile_wt'), n_cores= request.form.get('n_cores'),
                          primary_camera_pixels= request.form.get('primary_camera_pixels'), px_height= request.form.get('px_height'),
                          px_width= request.form.get('px_width'), ram= request.form.get('ram'), screen_height= request.form.get('screen_height'),
                          screen_width= request.form.get('screen_width'), talk_time= request.form.get('talk_time'), three_g= request.form.get('3g'),
                          touch_screen= request.form.get('touch_screen'), wifi= request.form.get('wifi'),
                          )
        pred_data_df = data.get_features_as_df()
        print(pred_data_df)
        
        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_data_df)
        return render_template('home.html', results= results[0])
    
print(5)
if __name__=="__main__":
    app.run(debug= True)  
    