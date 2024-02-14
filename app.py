from flask import Flask, request, render_template
import numpy as np
import pandas as pd

app = Flask(__name__)

# Route to home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predictdata',methods= ['GET',"POST"])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    
    else:
        return request.form.get('battery_power')

if __name__=="__main__":
    app.run(host="0.0.0.0")  