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


ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')
print(f'worspace details {ws}')

cluster_name = 'mlpipeline'

try:
	compute_target = ComputeTarget(workspace=ws, name=cluster_name)
	print('Found existing coumpute target')
except ComputeTargetException:
	print('Createing a new compute target...')
	compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_DS3_V2', max_nodes=2)
	compute_target = ComputeTarget.create(ws, cluster_name, compute_config)
	compute_target.wait_for_completion(show_output=True, min_node_count=None, timeout_in_minutes=20)

print('-'*101)
print('compute target created')
print('-'*101)

fastai_env = Environment.from_conda_specification(name="demand-new", file_path="env.yaml")

# fastai_env = Environment.from_pip_requirements(name="demand-new", file_path="requirements.txt")

fastai_env.register(workspace=ws)

args = ['--reg_rate', 0.1]
print('-'*101)
print('Environment variables created...')
print('-'*101)
docker_config = DockerConfiguration(use_docker=True)
print('running the script ....')
src = ScriptRunConfig(source_directory='.',
						script='train.py',
						arguments=args,
						compute_target=compute_target,
						environment=fastai_env,
						docker_runtime_config= docker_config
						)
                        
print('completed running the script...')
run = Experiment(workspace=ws, name='demand_forecasting').submit(src)
run.wait_for_completion(show_output=True)

file_names = run.get_file_names()


model_list = []
for file in file_names:
    if "model_" in file:
        model_list.append(file)

len(model_list)

#model_name = 'sarima_model.pkl'

print('-' * 100)
print('downloading the model')

for file in model_list:
    run.download_file(name=f'{file}', output_file_path="./docker_deploy/outputs/")
    print("*"*100)


