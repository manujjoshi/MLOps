from azureml.core.webservice import AksWebservice
from azureml.core import Environment, Workspace
from azureml.core.model import Model, InferenceConfig
from azureml.core.runconfig import DockerConfiguration 
from azureml.core.compute import ComputeTarget, AksCompute
from azureml.exceptions import ComputeTargetException

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')


aks_compute_name = "aksCompute"

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

aks_service_name = "akscomputeee41404613a"
model_scaler = Model(ws, "scaler_object", version=2)
model_classi = Model(ws,"Random_Forest_Regression")

aks_service = Model.deploy(ws, aks_service_name, [model_scaler,model_classi],inference_config,aks_config,aks_target)
aks_service.wait_for_deployment(show_output=True)