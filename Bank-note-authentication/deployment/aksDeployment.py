from azureml.core.webservice import AksWebservice
from azureml.core import Environment, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.core.runconfig import DockerConfiguration 
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

ws = Workspace.get(name='bgAMLdemo',
               subscription_id='a0093fb2-0cec-4cdd-b415-e44f05a01702',
               resource_group='bg-aml')

aks_compute_name = "taxi-trip-aks-compute"

try:
    aks_target = ComputeTarget(workspace=ws,name= aks_compute_name)
    print("Accessing Existing AKS Cluster")
except ComputeTargetException:
    prov_config = AksCompute.provisioning_configuration(vm_size="STANDARD_DS3_V2")
    aks_target = ComputeTarget.create(workspace=ws,name=aks_compute_name,provisioning_configuration=prov_config)
    aks_target.wait_for_completion(show_output=True)
    

env = Environment.get(ws,"env_taxi_trip",1)

aks_config = AksWebservice.deploy_configuration(cpu_cores=1,
                                               memory_gb=1,
                                               auth_enabled=False)
inference_config = InferenceConfig(entry_script="score.py",environment=env)

aks_service_name = "taxi_trip_prediction_aks"
model_scaler = Model(ws, "scaler_object", version=2)
model_classi = Model(ws,"RandomForestRegression")

aks_service = Model.deploy(ws, aks_service_name, [model_scaler,model_classi],inference_config,aks_config,aks_target)
aks_service.wait_for_deployment(show_output=True)