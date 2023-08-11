import numpy as np 
import pandas as pd
import itertools
import warnings
warnings.filterwarnings("ignore")
from math import sqrt
from multiprocessing import Pool
import joblib
from prophet import Prophet

from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Dataset
import joblib
from azureml.core.model import Model
from sklearn.metrics import mean_squared_error, mean_absolute_error

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')

df = pd.read_csv("data/train.csv")
# df.head()

df.rename(columns = {'date':'ds', 'sales':'y'}, inplace = True)
df = df.dropna()
df = df.loc[df.item <= 12]

print('libraries imported!')

# ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
# ws=Workspace.from_config(auth=ia)

print(ws)

# Get the experiment run context
run = Run.get_context()

run.log("shape", df.shape)

g_df = df.groupby(by=['store', 'item', 'ds'], as_index=False).sum('y')

#run.log_table("head", g_df.head())

import os 
os.makedirs('outputs', exist_ok=True)





# Define the function to train a model for a given group
def train_model(df):
    model = Prophet(
      interval_width=0.95,
      growth='linear',
      daily_seasonality=False,
      weekly_seasonality=True,
      yearly_seasonality=True,
      seasonality_mode='multiplicative'
      )
    model.fit(df)
    print('Reached till fit now dumping!!')
    joblib.dump(model, f'outputs/model_{df.store.values[0]}_{df.item.values[0]}')
    print(f'outputs/model_{df.store.values[0]}_{df.item.values[0]}')

    
    future_pd = model.make_future_dataframe(
      periods=0, 
      freq='d', 
      include_history=True
      )
    
    pred_df = model.predict(future_pd)

    mse = mean_squared_error(df.y.tolist(), pred_df.yhat.tolist())
    mae = mean_absolute_error(df.y.tolist(), pred_df.yhat.tolist())
    rmse = sqrt(mse)

    run.log(f"mae-{df.store.values[0]}-{df.item.values[0]}", mae)
    run.log(f"rmse-{df.store.values[0]}-{df.item.values[0]}", rmse)

    reg_model = Model.register(model_path=f"outputs/model_{df.store.values[0]}_{df.item.values[0]}",
                            model_name=f"model_{df.store.values[0]}_{df.item.values[0]}",
                            tags={'area': "demand Forecasting", 'type': "timeseries"},
                            description="demand forecasting",
                            workspace=ws)
    # return model
# Define the function to train models in parallel
def train_models_parallel(df, n_jobs=8):
    # Group the data by a specific column(s)
    grouped_data = df.groupby(by=['store','item'])

    for name, data in grouped_data:
      # models_dict[name] = train_model(data)
      train_model(data)


# Train models in parallel
train_models_parallel(g_df)

run.complete()