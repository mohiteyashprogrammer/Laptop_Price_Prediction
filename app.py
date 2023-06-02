from flask import Flask,request,render_template,jsonify
from src.pipline.prediction_pipline import CustomData,PredictPipline
import pandas as pd 
import os 
import sys 
import numpy as np 

application = Flask(__name__)
app = application

@app.route("/",methods = ["GET","POST"])
def predict_datapoint():
    if request.method == "GET":
        return render_template("form.html")

    else:
        data = CustomData(
            Company = request.form.get("Company")
            , TypeName = request.form.get("TypeName")
            , Inches = float(request.form.get("Inches"))
            , ScreenResolution = request.form.get("ScreenResolution")
            , Ram = int(request.form.get("Ram"))
            , OpSys = request.form.get("OpSys")
            , Weight = float(request.form.get("Weight"))
            , Touchscreen = int(request.form.get("Touchscreen"))
            , IPS_panel = int(request.form.get("IPS_panel"))
            , Cpu_brand = request.form.get("Cpu_brand")
            , HDD = int(request.form.get("HDD"))
            , SSD = int(request.form.get("SSD"))
            , Gpu_brand =  request.form.get("Cpu_brand")
        )

        final_data = data.get_data_as_data_frame()
        predict_pipline = PredictPipline()
        pred = np.exp(predict_pipline.Predict(final_data))

        result = round(pred[0],2)

        return render_template("form.html",final_result = "Your Laptop  Price Is: {}".format(result))

        
if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)  

