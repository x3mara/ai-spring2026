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
SEED = 442004



#%% main functions

def preprocess(df: pd.DataFrame, remove_nulls:bool) -> pd.DataFrame:
    if remove_nulls:
        for index,row in df.iterrows():
            if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
                df.at[index,'Outlet_Size'] = 'Missing'
            elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
                df.at[index,'Outlet_Size'] = 'Missing'

        # df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
        df = df.drop(columns=['Item_Weight'])

    df = df.drop(columns=['Item_Identifier'])
    df = pd.get_dummies(df, drop_first=True)
    return df

def get_train(do_preprocess = True, remove_nulls = True):
    df = pd.read_csv("train.csv")
    if do_preprocess:
        df = preprocess(df, remove_nulls=remove_nulls)
    columns = set(df.columns)
    X = df[sorted(list(columns - {'Y'}))]
    y = df['Y']
    return X, y

def get_test(remove_nulls=True):
    df = pd.read_csv("test.csv")
    df = preprocess(df,remove_nulls)
    return df[sorted(df.columns)]

def do_test(model, remove_nulls = True):
    prediction = model.predict(get_test(remove_nulls))
    submission = pd.DataFrame({
        'row_id': range(0, len(prediction)),
        'Y': prediction
    })
    submission.to_csv('output.csv',index=False)

def print_errors(model):
    predTrain = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, predTrain)
    predTest = model.predict(X_test)
    mae = mean_absolute_error(y_test, predTest)
    print(f"overfit mae = {train_mae}")
    print(f"mean absolute error = {float(mae)}")

#%%

X,y = get_train(remove_nulls=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, # ratio
    random_state = SEED)

#%%
len_reg = Pipeline([
    ("model", LinearRegression())
    ]) 
len_reg.fit(X_train, y_train)

print_errors(len_reg)
do_test(len_reg)

#%%
XG_reg = XGBRegressor(
        min_child_weight=3,
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        early_stopping_rounds=20,
        eval_metric = 'rmse',
        subsample=0.8,
        random_state=SEED
        )
XG_reg.fit(X_train,y_train,
           eval_set=[(X_test,y_test)],
           verbose=False)

print_errors(XG_reg)
do_test(XG_reg,remove_nulls=False)

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
