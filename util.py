#%%
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.metrics import mean_squared_error
from sklearn . linear_model import LogisticRegression
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score, KFold
from sklearn.compose import make_column_transformer, ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
SEED = 442004

def preprocess(df: pd.DataFrame, train_df:pd.DataFrame=None) -> pd.DataFrame:
    for index,row in df.iterrows():
        if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
            df.at[index,'Outlet_Size'] = 'Missing1'
        elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
            df.at[index,'Outlet_Size'] = 'Missing2'

    df['Outlet_Age'] = 2013 - df['Outlet_Est_Year'] 
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    df.loc[df['Item_Identifier'].str.startswith('NC'), 'Item_Fat_Content'] = 'Non-Edible'
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    df['Item_Category'] = df['Item_Category'].replace({'DR':'FD'})

    df['Item_Weight'] = df['Item_Weight'].fillna(
        df.groupby('Item_Identifier')['Item_Weight'].transform('mean')
    )
    df['Item_Weight'] = df['Item_Weight'].fillna(
        df.groupby('Item_Type')['Item_Weight'].transform('mean')
    )

    item_types = df['Item_Type'].unique()
    for item_type in item_types:
        median_vis = df.loc[df['Item_Type']==item_type,'Item_Visibility'].mean()
        df.loc[df['Item_Type']==item_type,'Item_Visibility'] = df.loc[df['Item_Type']==item_type,'Item_Visibility'].replace(0,median_vis)
        
    df['Item_Visibility_Log'] = np.log1p(df['Item_Visibility'])

    # df = df.drop(columns=['Item_Weight'])

    if 'Y' in df.columns:
        df["Mean_Y_by_ItemType"] = df.groupby("Item_Type")["Y"].transform("mean")
        df["Mean_Y_by_OutletType"] = df.groupby("Outlet_Type")["Y"].transform("mean")
    else:
        type_means = train_df.groupby('Item_Type')['Y'].mean()
        df['Mean_Y_by_ItemType'] = df['Item_Type'].map(type_means)
        df['Mean_Y_by_ItemType'] = df['Mean_Y_by_ItemType'].fillna(train_df['Y'].mean())
        type_means = train_df.groupby('Outlet_Type')['Y'].mean()
        df['Mean_Y_by_OutletType'] = df['Outlet_Type'].map(type_means)
        df['Mean_Y_by_OutletType'] = df['Mean_Y_by_OutletType'].fillna(train_df['Y'].mean())

    # MRP habal
    df['Outlet_Type_Avg_MRP'] = df.groupby('Outlet_Type')['Item_MRP'].transform('mean')
    df['Outlet_Avg_MRP'] = df.groupby('Outlet_Identifier')['Item_MRP'].transform('mean')
    df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']
    df['MRP_Squared'] = df['Item_MRP'] ** 2 
    df['Item_Price_Rank'] = df.groupby('Item_Category')['Item_MRP'].rank(pct=True)
    df['MRP_Outlet_Rank'] = df.groupby('Outlet_Identifier')['Item_MRP'].rank(pct=True)

    df['MRP_x_OutletType_Mean'] = df['Item_MRP'] * df['Outlet_Type_Avg_MRP']
    df['Visibility_x_MRP'] = df['Item_Visibility_Log'] * df['Item_MRP']
    df['OutletAge_x_MRP'] = df['Outlet_Age'] * df['Item_MRP']

    df = df.drop(columns=[
        'Item_Visibility',
        'Outlet_Est_Year'
    ])

    return df

def get_train():
    df = pd.read_csv("train.csv")
    df = preprocess(df)
    columns = set(df.columns)
    X = df[sorted(list(columns - {'Y'}))]
    y = df['Y']
    return X, y, df

def get_test(train_df:pd.DataFrame=None):
    df = pd.read_csv("test.csv")
    df = preprocess(df,train_df=train_df)
    return df[sorted(df.columns)]

def do_test(model, train_df:pd.DataFrame):
    prediction = model.predict(get_test(train_df=train_df))
    submission = pd.DataFrame({
        'row_id': range(0, len(prediction)),
        'Y': prediction
    })
    submission.to_csv('output.csv',index=False)

def print_errors(model, X_train, y_train, X_test, y_test):
    predTrain = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, predTrain)
    predTest = model.predict(X_test)
    mae = mean_absolute_error(y_test, predTest)
    print(f"overfit mae = {train_mae}")
    print(f"mean absolute error = {float(mae)}")
    # return predTest

# %%
