print('running the train.py file ..............................................')
print('-'*1000)
print()



from azureml.core import Workspace, Run
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core import Workspace, Dataset
print('loaded azureml dependencies!')

print('importing libraries ...')
# Import libraries
import numpy as np
import os
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pickle
import joblib

print('libraries imported!')

ia = InteractiveLoginAuthentication(tenant_id='e4e34038-ea1f-4882-b6e8-ccd776459ca0')
ws=Workspace.from_config(auth=ia)



print(ws)

# Get the experiment run context
run = Run.get_context()

# Set regularization hyperparameter
parser = argparse.ArgumentParser()
parser.add_argument('--reg_rate', type=float, dest='reg', default=0.01)
args = parser.parse_args()
reg = args.reg


df = pd.read_csv(r'data/house_data.csv')

columns = ['bedrooms', 'bathrooms', 'floors', 'yr_built', 'price']
df = df[columns]

X = df.iloc[:, 0:4]
y = df.iloc[:, 4:]


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

lr = LinearRegression()
lr.fit(X_train, y_train)

# calculate accuracy
y_hat = lr.predict(X_test)
acc = np.average(y_hat == y_test)
print('Accuracy:', acc)
run.log('Accuracy', np.float(acc))


os.makedirs('outputs', exist_ok=True)
joblib.dump(value=lr, filename='outputs/house_model.pkl')

run.complete()