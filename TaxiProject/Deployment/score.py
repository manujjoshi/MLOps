import json
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np
import joblib
import pandas as pd
from azureml.core.model import Model
from azureml.core import Workspace
# to fetch the models and store the models

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')
# def init():
#     global model_scaler, model_classi
#     model_scaler = joblib.load(Model.get_model_path('model_scaler',_workspace=ws))
#     model_classi = joblib.load(Model.get_model_path('model_classi',_workspace=ws))

def init():
    global model_scaler, model_classi

    scaler_path = Model.get_model_path("scaler_object",_workspace = ws)
    model_scaler = joblib.load(scaler_path)
    print("scaler load")
    
    model_path = Model.get_model_path("Random_Forest_Regression",_workspace = ws)
    model_classi = joblib.load(model_path)
    print("model load")

    # file_classi.close()  
    print('init running')
# data preprocessing for new data and prediction
def run(raw_data):
    in_data = pd.read_json(raw_data)
    in_data.drop(columns=["id","store_and_fwd_flag","pickup_datetime","dropoff_datetime" ],inplace=True)
    in_data = in_data.values
    in_data = model_scaler.transform(in_data)
    result = model_classi.predict(in_data)
    return {"calculated trip":result}
    