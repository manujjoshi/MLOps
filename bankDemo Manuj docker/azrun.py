from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import DockerConfiguration
import os
import pandas as pd



ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')

env = Environment.get(ws,"env_taxi_trip",1) 

compute_name = "mlopsdemo"

###############################

try:
    compute_cluster = ComputeTarget(ws,name=compute_name)
    print("using Existing cluster")
except ComputeTargetException:
    compute_config = AmlCompute.provision_config(vm_size = "Standard_NV6",
                                                 max_nodes = 1)
    compute_cluster = ComputeTarget.create(ws,compute_name,compute_config)
    compute_cluster.wait_for_completion(show_output=True)

###############################
    
docker_conf = DockerConfiguration(use_docker=True)
config_run = ScriptRunConfig(source_directory=".",
                             script="train.py",
                             compute_target=compute_name,
                             environment=env,
                             docker_runtime_config=docker_conf)

run = Experiment(ws,name="demo-bank").submit(config_run)
run.wait_for_completion(show_output=True)
print('-'*50)
print('-'*50)
print(os.getcwd())
print('-'*50)
print('-'*50)
print(os.listdir())
print('-'*50)
print('-'*50)
data = pd.read_csv('BankNote_Authentication.csv')
##data.to_csv('./dockerdeployment/data.csv')
##run.download_file(name='outputs/knn.pkl', output_file_path="./dockerdeployment/knn.pkl") 
print('-'*50)
print('-'*50)
print('Downloading the model!!')
print('-'*50)
print('-'*50)
run.download_file(name='outputs/knn.pkl',output_file_path='./dockerdeployment/knn.pkl')