# %% imports
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

SEED = 1234

#%% load csv file
df = pd.read_csv("train.csv")

#%% do preprocessing (Outlet_Size)
for index,row in df.iterrows():
    if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
        df.at[index,'Outlet_Size'] = 'Small'
    elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
        df.at[index,'Outlet_Size'] = 'Medium'
#%% do preprocessing (Item_Weight)
df = df.drop(columns=['Item_Weight'])

#%% guess imputation thing
freq = dict()
for index,row in df.iterrows():
    tmp = (row['Outlet_Location_Tier'],row['Outlet_Type'])
    if tmp not in freq.keys(): freq[tmp] = set()
    freq[tmp].add(row['Outlet_Size'])
for key,val in freq.items():
    print(f"{key}: {val}")
#%% more imputation!
freq = dict()
for index,row in df.iterrows():
    tmp = row['Item_Identifier']
    if tmp not in freq.keys(): freq[tmp] = []
    if str(row['Item_Weight']) != 'nan':
        freq[tmp] = list(row)
for key,val in freq.items():
    print(f"{key}: {sorted(val)}")

# %% 20 head
df.head(20)

#%% data splitting
columns = set(df.columns)
X = df[list(columns - {'Y'})]
y = df['Y']
