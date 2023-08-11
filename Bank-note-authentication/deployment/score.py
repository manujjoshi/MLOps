import json
from azureml.core.model import Model
import joblib
import numpy
from azureml.core import Workspace
from azureml.core import Workspace
ws = Workspace.get(name="nivesh-aml-workspace",
               subscription_id='a0093fb2-0cec-4cdd-b415-e44f05a01702',
               resource_group='nivesh-aml-demo')

def init():
    global loaded_model
    model_path = Model.get_model_path("knn_as_meta4",_workspace = ws)
    loaded_model = joblib.load(model_path)
    print("BankNote model loaded")
def run(raw_data):
    data = json.loads(raw_data)
    variance = data["variance"]
    skewness = data["skewness"]
    curtosis = data["curtosis"]    
    entropy = data["entropy"]
    Predicted_note = loaded_model.predict([[variance,skewness,curtosis,entropy]])[0]
    return Predicted_note
