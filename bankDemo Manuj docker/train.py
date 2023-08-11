from azureml.core import Workspace, Dataset, Experiment, Run
from azureml.core.model import Model as azm
import pandas as pd
import numpy as np
import pathlib
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import joblib
from joblib import dump
#accesing Workspace

#access the workspace

ws = Workspace.get(name='aml-DSTeam-RnD-001',
               subscription_id='c59b6c0a-0bc0-4b69-bd03-020b2171f742',
               resource_group='RG-AmlWS-DSTeam-RnD')

print(f"workspace details are {ws}")

#access dataset
# azureml-core of version 1.0.72 or higher is required
# azureml-dataprep[pandas] of version 1.1.34 or higher is required
# from azureml.core import Workspace, Dataset

# subscription_id = 'c59b6c0a-0bc0-4b69-bd03-020b2171f742'
# resource_group = 'RG-AKS-DSTeam-RnD'
# workspace_name = 'niveshamldemo'

# workspace = Workspace(subscription_id, resource_group, workspace_name)

# dataset = Dataset.get_by_name(workspace, name='niveshdemodataset')
# dataset = dataset.to_pandas_dataframe()

dataset = pd.read_csv('BankNote_Authentication.csv')



#####getting the run context####
run = Run.get_context()
run.tag("Description locally trained the obsence model! ")
#data prepreocessing

X = dataset.drop(["class"],axis = 1)
y = dataset["class"]  #dependent data

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.20)

#logging the shape of x_train,x_test,y_train,y_test
run.log("size of x_train", X_train.shape)
run.log("size of y_train", y_train.shape)

run.log("size of x_test", X_test.shape)
run.log("size of y_test", y_test.shape)

# creating meta models
c1 = LogisticRegression(solver='liblinear')
c2 = DecisionTreeClassifier()
c3 = RandomForestClassifier(max_depth=2)
c4 = KNeighborsClassifier(n_neighbors=5)
c1.fit(X_train,y_train)
c2.fit(X_train,y_train)
c3.fit(X_train,y_train)
c4.fit(X_train,y_train)
y_pred1 = c1.predict(X_test)
y_pred2 = c2.predict(X_test)
y_pred3 = c3.predict(X_test)
y_pred4 = c4.predict(X_test)

#accuracy 
acc1 = accuracy_score(y_pred1,y_test)
run.log("accuracy of model of logistic regression ",acc1)
acc2 = accuracy_score(y_pred2,y_test)
run.log("accuracy of model of decision tree is ",acc2)
acc3 = accuracy_score(y_pred3,y_test)
run.log("accuracy of model of randomforest is : ",acc3)
acc4 = accuracy_score(y_pred4,y_test)
run.log("accuracy of model of k-nearest neighbour is: ",acc4)


accuracy_list = {
    "c1": acc1,
    "c2" : acc2,
    "c3" : acc3,
    "c4": acc4
}

maximum_accuracy = max(acc1,acc2,acc3,acc4)

# list out keys and values separately
key_list = list(accuracy_list.keys())
val_list = list(accuracy_list.values())
 

position = val_list.index(maximum_accuracy)
name = key_list[position]

run.log("maximium accuracy is :",maximum_accuracy)
dump(c1,'./meta1.pkl')
dump(c2,'./meta2.pkl')
dump(c3,'./meta3.pkl')
dump(c4,'./meta4.pkl')



azm.register(workspace=ws,
             model_path="./meta1.pkl",
             model_name="logistic_regression_as_meta"
             )
azm.register(workspace=ws,
             model_path="./meta2.pkl",
             model_name="decision_tree_as_meta"
             )
azm.register(workspace=ws,
             model_path="./meta3.pkl",
             model_name="random_forest_as_meta"
             )
azm.register(workspace=ws,
             model_path="./meta4.pkl",
             model_name="knn_as_meta"
             )

os.makedirs('outputs', exist_ok=True)
joblib.dump(value=c4, filename='outputs/knn.pkl')

run.complete()