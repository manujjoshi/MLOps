from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import DockerConfiguration



ws = Workspace.get(name="niveshamldemo",
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AKS-DSTeam-RnD')


env = Environment.get(ws,"env1",1)

compute_name = "democomputeinstance"
try:
    compute_cluster = ComputeTarget(ws,name=compute_name)
    print("using Existing cluster")
except ComputeTargetException:
    compute_config = AmlCompute.provision_config(vm_size = "Standard_NV6",
                                                 max_nodes = 1)
    compute_cluster = ComputeTarget.create(ws,compute_name,compute_config)
    compute_cluster.wait_for_completion(show_output=True)
    
docker_conf = DockerConfiguration(use_docker=True)
config_run = ScriptRunConfig(source_directory=".",
                             script="train.py",
                             compute_target=compute_cluster,
                             environment=env,
                             docker_runtime_config=docker_conf)

run = Experiment(ws,name="demo-bank").submit(config_run)
run.wait_for_completion(show_output=True)