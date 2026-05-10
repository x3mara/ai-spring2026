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
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor, StackingRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
from fast_ensemble import StackingTransformer, CatBoostRegressorWrapper
from sklearn.preprocessing import LabelEncoder
import sklearn
sklearn.set_config(enable_metadata_routing=True)
SEED = 442004



#%% main functions

def preprocess(df: pd.DataFrame, remove_nulls:bool, train_df:pd.DataFrame=None) -> pd.DataFrame:
    if remove_nulls:
        for index,row in df.iterrows():
            if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
                df.at[index,'Outlet_Size'] = 'Missing'
            elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
                df.at[index,'Outlet_Size'] = 'Missing'

        #df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())
    
    df['Outlet_Age'] = 2026 - df['Outlet_Est_Year'] 
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})

    item_types = df['Item_Type'].unique()
    for item_type in item_types:
        median_vis = df.loc[df['Item_Type']==item_type,'Item_Visibility'].median()
        median_weight = df.loc[df['Item_Type']==item_type,'Item_Weight'].median()
        df.loc[df['Item_Type']==item_type,'Item_Visibility'] = df.loc[df['Item_Type']==item_type,'Item_Visibility'].replace(0,median_vis)
        df.loc[df['Item_Type']==item_type,'Item_Weight'] = df.loc[df['Item_Type']==item_type,'Item_Weight'].fillna(median_weight)

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

    df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']

    df = df.drop(columns=['Item_Identifier','Outlet_Identifier','Item_Visibility','Outlet_Est_Year'])
    #df = pd.get_dummies(df, drop_first=True)
    return df

def get_train(do_preprocess = True, remove_nulls = True):
    df = pd.read_csv("train.csv")
    if do_preprocess:
        df = preprocess(df, remove_nulls=remove_nulls)
    columns = set(df.columns)
    X = df[sorted(list(columns - {'Y'}))]
    y = df['Y']
    return X, y, df

def get_test(remove_nulls=True,train_df:pd.DataFrame=None):
    df = pd.read_csv("test.csv")
    df = preprocess(df,remove_nulls,train_df=train_df)
    return df[sorted(df.columns)]

def do_test(model, train_df:pd.DataFrame, remove_nulls = True):
    prediction = model.predict(get_test(remove_nulls,train_df=train_df))
    submission = pd.DataFrame({
        'row_id': range(0, len(prediction)),
        'Y': prediction
    })
    submission.to_csv('output.csv',index=False)

def print_errors(model, X_train, X_test, y_train, y_test):
    predTrain = model.predict(X_train)
    train_mae = mean_absolute_error(y_train, predTrain)
    predTest = model.predict(X_test)
    mae = mean_absolute_error(y_test, predTest)
    print(f"overfit mae = {train_mae}")
    print(f"mean absolute error = {float(mae)}")
    return predTest

def do_encoding(X):
    return pd.get_dummies(X, drop_first=True)

#%%

X,y,train_df = get_train(remove_nulls=True)

from sklearn.model_selection import KFold

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.8, random_state = SEED)

# %%
stringcols = list(X_train.select_dtypes(include='str').columns)
stringcolidx = [X_train.columns.get_loc(col) for col in stringcols]

cat_reg = CatBoostRegressor(
    cat_features=stringcols,
    iterations=1500,
    learning_rate=0.05,
    early_stopping_rounds=20,
    use_best_model=True,
    depth=5,
    l2_leaf_reg=3,
    loss_function='MAE'
)
cat_reg.fit(X_train,y_train,
    eval_set=(X_test,y_test),
    verbose=False)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.5, random_state = SEED)

xxx=print_errors(cat_reg,X_train,X_test,y_train,y_test)
do_test(cat_reg,train_df=train_df, remove_nulls=True)

#%%
len_reg = Pipeline([
    ("dummies", FunctionTransformer(do_encoding,validate=False)),
    ("model", LinearRegression())
    ]) 
len_reg.fit(X_train, y_train)

yyy=print_errors(len_reg,X_train,X_test,y_train,y_test)
# do_test(len_reg,train_df=train_df)

#%%
stacking = StackingRegressor(
    estimators=[
        ('catboost',cat_reg),
        ('linear', len_reg)
    ],
    final_estimator=LinearRegression(),
    cv=5
)
stacking.fit(X_train,y_train)
print_errors(stacking)
do_test(stacking,train_df=train_df, remove_nulls=True)

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

xxx=print_errors(XG_reg)
do_test(XG_reg,remove_nulls=False)

# %%
rf = RandomForestRegressor(n_estimators=900, random_state=442004)
rf.fit(X_train, y_train)
zzz =rf_pred = rf.predict(X_test)


# %%

finale =(.99*xxx) +(.01*yyy)
mae=mean_absolute_error(y_test, finale)
print (mae)

# %%
# %%
# %%
# %%
# %%
print(X_train.info())
# %%
X_train.isnull().sum()
# %%
print(stringcols)
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