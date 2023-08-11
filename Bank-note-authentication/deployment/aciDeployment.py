from azureml.core.webservice import AciWebservice
from azureml.core import Environment, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.core.runconfig import DockerConfiguration 

ws = Workspace.get(name='bgAMLdemo',
               subscription_id='a0093fb2-0cec-4cdd-b415-e44f05a01702',
               resource_group='bg-aml')

env = Environment.get(ws,"env_taxi_trip",1)

aci_config = AciWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               auth_enabled=False)
inference_config = InferenceConfig(entry_script="score.py",environment=env)
aci_service_name = "taxitrippredictionaci"
model_scaler = Model(ws, "scaler_object", version=2)
model_classi = Model(ws,"RandomForestRegression")

aci_service = Model.deploy(ws, aci_service_name, [model_scaler,model_classi],inference_config,aci_config,overwrite=True)
aci_service.wait_for_deployment(show_output=True)