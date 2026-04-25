# %% imports
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn . linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from sklearn.preprocessing import LabelEncoder
SEED = 1234



#%% main functions

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    for index,row in df.iterrows():
        if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
            df.at[index,'Outlet_Size'] = 'Small'
        elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
            df.at[index,'Outlet_Size'] = 'Medium'

    # df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
    df = df.drop(columns=['Item_Weight'])

    df = df.drop(columns=['Item_Identifier'])
    df = pd.get_dummies(df, drop_first=True)
    return df

def get_train():
    df = pd.read_csv("train.csv")
    df = preprocess(df)
    columns = set(df.columns)
    X = df[sorted(list(columns - {'Y'}))]
    y = df['Y']
    return X, y

def get_test():
    df = pd.read_csv("test.csv")
    df = preprocess(df)
    return df[sorted(df.columns)]

def do_test(model):
    prediction = model.predict(get_test())
    submission = pd.DataFrame({
        'row_id': range(1, len(prediction) + 1),
        'prediction': prediction
    })
    submission.to_csv('output.csv',index=False)


#%%

X,y = get_train()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, # ratio
    random_state = SEED)

len_reg = Pipeline([
    ("model", LinearRegression())
    ]) 
len_reg.fit(X_train, y_train)
len_predTest = len_reg.predict(X_test)
len_predTrain = len_reg.predict(X_train) 
mae = mean_absolute_error(y_train, len_predTrain)
mape = np.mean(np.abs((y_test - len_predTest) / y_test)) * 100
print(mae)
print(mape)

#%%
do_test(len_reg)

# commented code that is too valuable to remove
# #%% guess imputation thing
# freq = dict()
# for index,row in df.iterrows():
#     tmp = (row['Outlet_Location_Tier'],row['Outlet_Type'])
#     if tmp not in freq.keys(): freq[tmp] = set()
#     freq[tmp].add(row['Outlet_Size'])
# for key,val in freq.items():
#     print(f"{key}: {val}")
# #%% more imputation!
# freq = dict()
# for index,row in df.iterrows():
#     tmp = row['Item_Identifier']
#     if tmp not in freq.keys(): freq[tmp] = []
#     if str(row['Item_Weight']) != 'nan':
#         freq[tmp] = list(row)
# for key,val in freq.items():
#     print(f"{key}: {sorted(val)}")
# # %% 20 head
# df.head(20)

# # %%
# # by applying this => missing is random
# df.groupby('Item_Type')['Item_Weight'].apply(lambda x: x.isna().mean()*100) 
# # %%

# %%
