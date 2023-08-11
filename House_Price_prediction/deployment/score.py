    
import pickle
import json
import numpy as np
from azureml.core.model import Model
import joblib



def init():
    global model

    # load the model from file into a global object
    print("loading the model .....")
    model_path = Model.get_model_path(model_name="house_model.pkl")
    model = joblib.load(model_path)
    
def run(raw_data):
#     try:
    data = json.loads(raw_data)["data"]
    val1=data["bedrooms"]
    val2=data["bathrooms"]
    val3=data["floors"]
    val4=data["yr_built"]
    print('ran')

    arr = np.array([val1, val2, val3, val4])
    arr = arr.astype(np.float64)
    pred = model.predict([arr])
    return json.dumps({"house result": pred[0].tolist()})



# {"data":{"bedrooms":2,"bathrooms":2,"floors":1,"yr_built":2}}
