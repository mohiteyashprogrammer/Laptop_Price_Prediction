import os 
import sys
import pandas as pd 
import numpy as np
from dataclasses import dataclass
from src.logger import logging
from src.exception import CustomException

from src.utils import load_object

class PredictPipline:
    def __init__(self):
        pass

    def Predict(self,features):
        try:
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            Company:str,
            TypeName:str,
            Inches:float,
            ScreenResolution:str,
            Ram:int,
            OpSys:str,
            Weight:float,
            Touchscreen:int,
            IPS_panel:int,
            Cpu_brand:str,
            HDD:int,
            SSD:int,
            Gpu_brand:str):

        self.Company = Company
        self.TypeName = TypeName
        self.Inches = Inches
        self.ScreenResolution = ScreenResolution
        self.Ram = Ram
        self.OpSys = OpSys
        self.Weight = Weight
        self.Touchscreen = Touchscreen
        self.IPS_panel = IPS_panel
        self.Cpu_brand = Cpu_brand
        self.HDD = HDD
        self.SSD = SSD
        self.Gpu_brand = Gpu_brand

    
    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Company": [self.Company],
                "TypeName": [self.TypeName],
                "Inches": [self.Inches],
                "ScreenResolution": [self.ScreenResolution],
                "Ram": [self.Ram],
                "OpSys": [self.OpSys],
                "Weight": [self.Weight],
                "Touchscreen": [self.Touchscreen],
                "IPS_panel": [self.IPS_panel],
                "Cpu_brand": [self.Cpu_brand],
                "HDD": [self.HDD],
                "SSD": [self.SSD],
                "Gpu_brand": [self.Gpu_brand]
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Prediction Pipline")
            raise CustomException(e, sys)