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
from lightgbm import LGBMRegressor
#from fast_ensemble import StackingTransformer, CatBoostRegressorWrapper
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin
sklearn.set_config(enable_metadata_routing=True)
SEED = 442004

#%% main functions

def preprocess(df: pd.DataFrame, train_df:pd.DataFrame=None) -> pd.DataFrame:
    for index,row in df.iterrows():
        if row['Outlet_Location_Tier'] == 'Tier 2' and row['Outlet_Type'] == 'Supermarket Type1':
            df.at[index,'Outlet_Size'] = 'Missing1'
        elif row['Outlet_Location_Tier'] == 'Tier 3' and row['Outlet_Type'] == 'Grocery Store':
            df.at[index,'Outlet_Size'] = 'Missing2'

    df['Outlet_Age'] = 2026 - df['Outlet_Est_Year'] 
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({'LF': 'Low Fat', 'reg': 'Regular', 'low fat': 'Low Fat'})
    df['Item_Category'] = df['Item_Identifier'].str[:2]
    df['MRP_Bucket'] = pd.cut(df['Item_MRP'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])

    item_types = df['Item_Type'].unique()
    for item_type in item_types:
        median_vis = df.loc[df['Item_Type']==item_type,'Item_Visibility'].mean()
        median_weight = df.loc[df['Item_Type']==item_type,'Item_Weight'].mean()
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

    # df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']

    df = df.drop(columns=['Item_Identifier','Outlet_Identifier','Item_Visibility','Outlet_Est_Year'])

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

def do_encoding(X):
    return pd.get_dummies(X, drop_first=True)

#%% initialize

X,y,train_df = get_train()
cat_cols = X.select_dtypes(include=["object","string","category"]).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)

encoder = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'),cat_cols)
    ],
    remainder='passthrough'
)

#%% catboost

class CatBoostWrapper(CatBoostRegressor):
    def __init__(self, **kwargs):
        self.cat_features = cat_cols
        self.eval_set = (X_test, y_test)
        super().__init__(**kwargs)
    def fit(self, X, y):
        return super().fit(X, y,
                              cat_features=self.cat_features,
                              eval_set=self.eval_set)
    def predict(self, X):
        return super().predict(X)

catboost = CatBoostWrapper(
    iterations=1000,
    learning_rate=0.02,
    early_stopping_rounds=20,
    use_best_model=True,
    depth=2,
    bagging_temperature=0.5,
    l2_leaf_reg=3,
    loss_function='MAE',
    verbose=False
)
catboost.fit(X_train, y_train)
print_errors(catboost, X_train, y_train, X_test, y_test)
# do_test(catboost, train_df)

#%% xgb
class XGBRegressorWrapper(XGBRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def fit(self, X, y, **fit_params):
        X_test_enc = encoder.transform(X_test)
        self.eval_set = [(X_test_enc, y_test)]
        return super().fit(X, y,
                              eval_set=self.eval_set,
                              **fit_params,
                              verbose=False)
    def predict(self, X):
        return super().predict(X)


xgb = Pipeline(
    steps=[
        ('enc', encoder),
        ('model',XGBRegressorWrapper(
            min_child_weight=1,
            n_estimators=300,
            learning_rate=0.02,
            max_depth=2,
            objective='reg:absoluteerror',
            early_stopping_rounds=30,
            random_state=SEED
        ))
    ]
)
xgb.fit(X_train, y_train)
print_errors(xgb, X_train, y_train, X_test, y_test)
# do_test(xgb, train_df)

#%% linear dumb
linear = Pipeline(
    steps=[
        ('enc', encoder),
        ('model',LinearRegression())
    ]
)
linear.fit(X_train, y_train)
print_errors(linear, X_train, y_train, X_test, y_test)

#%% LGB

lgb = Pipeline(
    steps=[
        ('enc', encoder),
        ('model',LGBMRegressor(
            n_estimators=300,
            learning_rate=0.025,
            max_depth=2,
            # num_leaves=15,
            objective='mae',
            random_state=42,
            verbose=-1
        ))
    ]
)
lgb.fit(X_train, y_train)
print_errors(lgb, X_train, y_train, X_test, y_test)

#%%
lgb.fit(X,y)
do_test(lgb, train_df)

#%%

final = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('cat', catboost),
        ('lgb', lgb),
        ('linear', linear)
    ],
    final_estimator=XGBRegressor(
        n_estimators=899,
        learning_rate=0.02,
        max_depth=2,
        # subsample=0.8,
        # colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    ),
    cv=5
)
final.fit(X_train,y_train)
print_errors(final,X_train,y_train,X_test,y_test)

#%%
final.fit(X,y)
do_test(final, train_df)

# %%
