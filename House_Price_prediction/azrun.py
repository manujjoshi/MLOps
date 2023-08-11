import azureml
import os
from azureml.core import Experiment
from azureml.core import Environment
from azureml.core import Workspace, Run
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.exceptions import ComputeTargetException
from azureml.core.runconfig import DockerConfiguration
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import ScriptRunConfig
# from azureml.core.model import Model

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws = Workspace.from_config(auth=ia)
print(f'worspace details {ws}')
# def registerModel(model_path, model_name):
# 	Model.register(workspace=ws, model_path=model_path, model_name=model_name)
cluster_name = 'compute-cluster'

try:
	compute_target = ComputeTarget(workspace=ws, name=cluster_name)
	print('Found existing coumpute target')
except ComputeTargetException:
	print('Createing a new compute target...')
	compute_config = AmlCompute.provisioning_configuration(vm_size='Standard_D2as_v4', max_nodes=2)
	compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
	compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

print('-'*101)
print('compute target created')
print('-'*101)


# demo_env = Environment(ws, name="newenv")
# demo_env = Environment(name="house-newenv1")


# print('loading the conda dependencies..')
# for pip_package in ["joblib","scikit-learn","pandas","azureml-sdk", "numpy", "argparse"]:
#     demo_env.python.conda_dependencies.add_pip_package(pip_package)
# 

fastai_env = Environment.from_pip_requirements(name="house-env-docker", file_path="requirements.txt")

# docker_config = DockerConfiguration(use_docker=True)
# fastai_env.docker.base_image = "acrdsteamrnd001.azurecr.io/rahul/house-image:v1"
# fastai_env.python.user_managed_dependencies = True
# fastai_env.docker.base_image_registry.address = "acrdsteamrnd001.azurecr.io"
# fastai_env.docker.base_image_registry.username = "acrDSTeamRnD001"
# fastai_env.docker.base_image_registry.password = "X9/hByZAQTOp/MlgEhwL7kqI1lWl6WAM"
# fastai_env.register(workspace=ws)



args = ['--reg_rate', 0.1]
print('-'*101)
print('Environment variables created...')
print('-'*101)
docker_config = DockerConfiguration(use_docker=True)
print('running the script my friend ....')
src = ScriptRunConfig(source_directory='.',
						script='train.py',
						arguments=args,
						compute_target=compute_target,
						environment=fastai_env,
						docker_runtime_config= docker_config
						)
print('completed running the script...')
run = Experiment(workspace=ws, name='house-workspace').submit(src)
run.wait_for_completion(show_output=True)

model_name = 'house_model.pkl'

print('-' * 100)
print('downloading hte model')

run.download_file(name='outputs/house_model.pkl', output_file_path="./docker_deploy/house_model.pkl")
print("*"*100)

print('registering the model...')
if run.get_status() == 'Completed':
	model = run.register_model(
    		model_name=model_name,
        	model_path=f'outputs/{model_name}'
        )
	print('model registered!')

print('-'*101)
print('compute target created')
print('-'*101)

# fastai_env = Environment.from_dockerfile(name="house-env-dockerfile", dockerfile="./docker_deploy/Dockerfile")
# fastai_env.register(ws)

# fast_ai = Environment("house-env-dockerfile-new")
# fast_ai.docker.base_dockerfile = './docker_deploy/Dockerfile' # path to your dockerfile
# # optional
# fast_ai.python.user_managed_dependencies = True
# # env.python.interpreter_path = '/opt/miniconda/envs/example/bin/python'


# print("environment build completed!")

print('Hey completed the azrun.py MAN ..............')




# import os
# import azureml
# from azureml.core import Workspace, ScriptRunConfig
# from azureml.core.authentication import ServicePrincipalAuthentication
# from azureml.core import Experiment, Environment, Workspace, Run
# from azureml.core.compute import ComputeTarget, AmlCompute
# from azureml.core.compute_target import ComputeTargetException
# from azureml.core.authentication import InteractiveLoginAuthentication
# from azureml.core.runconfig import DockerConfiguration
# from azureml.core.model import Model
# from threading import Thread
# from azureml.core.authentication import ServicePrincipalAuthentication

# print("Welcome")
# ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
# ws = Workspace.from_config(auth=ia)

# print(ws)

# cluster_name = "mlops-compute"

# try:
#     compute_target=ComputeTarget(workspace=ws, name=cluster_name)
#     print("found existing target")
# except ComputeTargetException:
#     print("creating a new compute target")
#     compute_config = AmlCompute.provisioning_configuration(vm_size="STANDARD_DS3_V2", max_nodes=2)
#     compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
#     compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

# print("creating the environemnt...")
# # demo_env = Environment.get(workspace=ws, name="newenv")
# demo_env = Environment(name="newenv")
# demo_env.from_pip_requirements("newenv", "requirements.txt")
# docker_config = DockerConfiguration(use_docker=True)

# print("environment created !!!")

# print("running the script...")
# src = ScriptRunConfig(source_directory=".", script='train.py',
#                                 compute_target=compute_target,
#                                 environment=demo_env,
#                                 docker_runtime_config=docker_config)

# print('script run completed')

# print("/n")
# print("*"*100)
# print("/n")

# print("setting the experiment...")
# print("and submitting the training file to experiment run")
# run = Experiment(workspace=ws, name="mlops_experiment").submit(src)

# run.wait_for_completion(show_output=True)

# print("run completed!!!")
# model_name = 'diabetes_model.pkl'

# print('registering the model...')
# if run.get_status() == 'Completed':
# 	model = run.register_model(
#     		model_name=model_name,
#         	model_path=f'outputs/{model_name}'
#         )
# 	print('model registered!')

# print('Experiment completed ..............')

# rmxl6x3leckuifixommw4qb7p5v6z5feyqiazmrvfsdurf7ikxca