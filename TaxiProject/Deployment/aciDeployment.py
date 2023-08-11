from azureml.core.webservice import AciWebservice
from azureml.core import Environment, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.core.runconfig import DockerConfiguration 

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')

env = Environment.get(ws,"env_taxi_trip",1)

aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               auth_enabled=False)
inference_config = InferenceConfig(entry_script="score.py",environment=env)
aci_service_name = "taxitrippredictionaci"
model_scaler = Model(ws, "scaler_object", version=2)
model_classi = Model(ws,"Random_Forest_Regression")

aci_service = Model.deploy(ws, aci_service_name, [model_scaler,model_classi],inference_config,aci_config,overwrite=True)
aci_service.wait_for_deployment(show_output=True)

