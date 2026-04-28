# %%
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
from sklearn.preprocessing import LabelEncoder
import sklearn
sklearn.set_config(enable_metadata_routing=True)
SEED = 442004


# %%
#analyzing the data 
df=pd.read_csv("train.csv")
# %%
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
df["MRP_Outlet_Type"] = df["Item_MRP"].astype(str) + "_" + df["Outlet_Type"]
df['Price_Per_Weight'] = df['Item_MRP'] / df['Item_Weight']
df['MRP_Bucket'] = pd.cut(df['Item_MRP'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])
df['Outlet_Size'] = df['Outlet_Size'].fillna('Missing')
df['Outlet_Type_Size'] = df['Outlet_Type'] + '_' + df['Outlet_Size']
df['Outlet_Age'] = 2026 - df['Outlet_Est_Year'] 
df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
df['Item_Category'] = df['Item_Identifier'].str[:2]
df['Item_Category'] = df['Item_Category'].map({
    'FD':'Food',
    'DR':'Drinks',
    'NC':'Non-Consumable'
})
df["Item_Outlet_Type"] = df["Item_Type"] + "_" + df["Outlet_Type"]

df["Item_Visibility"] = np.where(
    df["Item_Visibility"] == 0,
    df.groupby("Item_Identifier")["Item_Visibility"].transform("mean"),
    df["Item_Visibility"]
)
df["Outlet_Size"] = df["Outlet_Size"].fillna("Missing")

df.drop(["Item_Identifier", "Outlet_Est_Year"], axis=1, inplace=True)
df['Item_Visibility_Log'] = np.log1p(df['Item_Visibility'])


# %%


df.info()
df.describe()
# %%
for i in range(0, 11):
    if df.iloc[:, i].dtype!= 'str':
        print(df.columns[i], df.iloc[:, i].skew())

 
# %%
for i in range(0, 11):
    if df.iloc[:, i].dtype== 'str':
     print(df.columns[i], df.iloc[:, i].value_counts())

# %%
df.apply(lambda x: len(x.unique()))

# %%
from sklearn.model_selection import KFold

X = df.drop("Y", axis=1)
y = df["Y"]

stringcols = X.select_dtypes(include=["object"]).columns.tolist()
stringcols.append('MRP_Bucket')
stringcols.append('Item_Outlet_Type')

kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

scores = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X)):

    X_train, X_val = X.iloc[train_idx].copy(), X.iloc[val_idx].copy()
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
   
    model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    early_stopping_rounds=20,
    use_best_model=True,
    depth=2,
    bagging_temperature=0.5,
    l2_leaf_reg=3,
    loss_function='MAE',
    verbose=False
    )
    
    model.fit(
        X_train, y_train,
        cat_features=stringcols,
        eval_set=(X_val, y_val),
        use_best_model=True,
        early_stopping_rounds=50
        )

    preds = model.predict(X_val)

    mae = mean_absolute_error(y_val, preds)
    scores.append(mae)

    print(f"Fold {fold+1} MAE: {mae}")

# final score
print("\nFinal CV MAE:", np.mean(scores))

Final_model = CatBoostRegressor(
iterations=1000,
learning_rate=0.05,
depth=2,
bagging_temperature=0.5,
l2_leaf_reg=3,
loss_function='MAE',
verbose=False
)

Final_model.fit(
    X, y,
    cat_features=stringcols,

    )
preds=Final_model.predict(X)

# %%
dftest=pd.read_csv('test.csv')
dftest['Item_Fat_Content'] = dftest['Item_Fat_Content'].replace({
        'LF': 'Low Fat',
        'low fat': 'Low Fat',
        'reg': 'Regular'
    })
dftest["Item_Outlet_Type"] = dftest["Item_Type"] + "_" + dftest["Outlet_Type"]
dftest['Price_Per_Weight'] = dftest['Item_MRP'] / dftest['Item_Weight']
dftest['MRP_Bucket'] = pd.cut(dftest['Item_MRP'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])
dftest['Outlet_Size'] = dftest['Outlet_Size'].fillna('Missing')
dftest['Outlet_Type_Size'] = dftest['Outlet_Type'] + '_' + dftest['Outlet_Size']
dftest["Outlet_Age"] = 2026 - dftest["Outlet_Est_Year"]
dftest['Item_Fat_Content'] = dftest['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
dftest['Item_Category'] = dftest['Item_Identifier'].str[:2]
dftest['Item_Category'] = dftest['Item_Category'].map({
    'FD':'Food',
    'DR':'Drinks',
    'NC':'Non-Consumable'
})


mean_vis = df.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
dftest.loc[dftest["Item_Visibility"] == 0, "Item_Visibility"] = mean_vis[dftest["Item_Visibility"] == 0]
dftest["Item_Visibility"] = dftest["Item_Visibility"].fillna(
    dftest.groupby("Item_Identifier")["Item_Visibility"].transform("mean")
)
train_mean_vis = df.groupby("Item_Identifier")["Item_Visibility"].mean()

dftest["Item_Visibility_MeanRatio"] = (
    dftest["Item_Visibility"] /
    dftest["Item_Identifier"].map(train_mean_vis)
)
dftest["Outlet_Size"] = dftest["Outlet_Size"].fillna("Missing")
dftest["MRP_Outlet_Type"] = dftest["Item_MRP"].astype(str) + "_" + dftest["Outlet_Type"]

dftest.drop(["Item_Identifier", "Outlet_Est_Year"], axis=1, inplace=True)
dftest['Item_Visibility_Log'] = np.log1p(dftest['Item_Visibility'])
prediction = Final_model.predict(dftest)
submission = pd.DataFrame({
        'row_id': range(0, len(prediction)),
        'Y': prediction
    })
submission.to_csv('output.csv',index=False)

# %%
df.describe()
# %%
