from azureml.core import Workspace, Dataset, Environment, Experiment, ScriptRunConfig, Run
from azureml.core.model import Model as azm


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
import math
from joblib import dump

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')

run = Run.get_context()

data = pd.read_csv('train_new.csv')
#data = Dataset.get_by_name("train").to_pandas_dataframe
data_shape = data.shape
print(data_shape)
run.log("raw data size",data_shape)


data.drop(columns=["id","store_and_fwd_flag","pickup_datetime","dropoff_datetime" ],inplace=True)

X = data.iloc[:,:-1].values
y = data.iloc[:,-1].values

print(len(X),len(y))
run.log("size of Features", len(X))
run.log("size of taregt", len(y))

X_train, X_test, y_train, y_test  = train_test_split(X,y,train_size=0.8)
print(len(X_train),len(y_train))
print(len(X_test),len(y_test))
run.log("x_train",X_train)
run.log("y_train",y_train)
run.log("x_test",X_test)
run.log("y_test",y_test)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

dump(sc,'./scaler')
azm.register(workspace=ws,
            model_path='./scaler',
            model_name = "scaler_object")

m1 = LinearRegression()
m2 = DecisionTreeRegressor(max_depth=4)
m3 = RandomForestRegressor(max_depth=4, n_estimators=100)

models = {
    "linear_regression":m1,
    "Decision_Tree_Regression":m2,
    "Random_Forest_Regression":m3
}

def train_model(model,name,pos):
    model.fit(X_train,y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test,y_pred)
    mse = mean_squared_error(y_test,y_pred)
    rmse = math.sqrt(mse)
    print("-----model is trained and result is captured-----")
    dump(model,f"./model{pos}")
    azm.register(workspace=ws,
                model_path=f"./model{pos}",
                model_name=name)
    return r2,mse,rmse,name,model


result = []
i=1
for name,model in models.items():
    result.append(train_model(model,name,i))
    i = i+1
print("model trainned and registered")

final_results = pd.DataFrame(result,columns=["R2_Score", "mean_square_error", "rmse", "model","model_object"])
final_results.drop(columns="model_object")
run.log("result",final_results)

run.complete()