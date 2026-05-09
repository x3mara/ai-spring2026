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
#from fast_ensemble import StackingTransformer, CatBoostRegressorWrapper
from sklearn.preprocessing import LabelEncoder
import sklearn
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin
sklearn.set_config(enable_metadata_routing=True)
SEED = 442004

#%% main functions

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
    # df['MRP_Bucket'] = pd.cut(df['Item_MRP'], bins=4, labels=['Low', 'Medium', 'High', 'Premium'])

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
    m = 10
    
    if 'Y' in df.columns:
        global_mean = df['Y'].mean()
        item_stats = df.groupby('Item_Type')['Y'].agg(['count', 'mean'])
        item_smoothed = (item_stats['count'] * item_stats['mean'] + m * global_mean) / (item_stats['count'] + m)
        df['Mean_Y_by_ItemType'] = df['Item_Type'].map(item_smoothed)
        outlet_stats = df.groupby('Outlet_Type')['Y'].agg(['count', 'mean'])
        outlet_smoothed = (outlet_stats['count'] * outlet_stats['mean'] + m * global_mean) / (outlet_stats['count'] + m)
        df['Mean_Y_by_OutletType'] = df['Outlet_Type'].map(outlet_smoothed)

    elif train_df is not None:
        global_mean = train_df['Y'].mean()
        item_stats = train_df.groupby('Item_Type')['Y'].agg(['count', 'mean'])
        item_smoothed = (item_stats['count'] * item_stats['mean'] + m * global_mean) / (item_stats['count'] + m)
        df['Mean_Y_by_ItemType'] = df['Item_Type'].map(item_smoothed).fillna(global_mean)
        outlet_stats = train_df.groupby('Outlet_Type')['Y'].agg(['count', 'mean'])
        outlet_smoothed = (outlet_stats['count'] * outlet_stats['mean'] + m * global_mean) / (outlet_stats['count'] + m)
        df['Mean_Y_by_OutletType'] = df['Outlet_Type'].map(outlet_smoothed).fillna(global_mean)
    
    df['Price_Per_Unit_Weight'] = df['Item_MRP'] / df['Item_Weight']

    df = df.drop(columns=['Item_Visibility','Outlet_Est_Year'])

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
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.to_list()

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

cat_selected_cols = cat_cols + ['Item_MRP']

class CatBoostWrapper(CatBoostRegressor):
    def __init__(self, **kwargs):
        self.eval_set = (X_test[cat_selected_cols], y_test)
        self.selected_cols = cat_selected_cols
        super().__init__(**kwargs)
    def get_cat_features(self, tX):
        return tX.select_dtypes(include=["object","string","category"]).columns.tolist()
    def fit(self, X, y):
        tX = X[self.selected_cols]
        return super().fit(tX, y,
                              cat_features=self.get_cat_features(tX),
                              eval_set=self.eval_set)
    def predict(self, X):
        tX = X[self.selected_cols]
        return super().predict(tX)
    
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

#%% get catboost importance
importance_df = pd.DataFrame({
    'feature': catboost.feature_names_,
    'importance': catboost.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(100))

#%% xgb

xgb_selected_cols = numeric_cols

xgb_encoder = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X[xgb_selected_cols].select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'),X[xgb_selected_cols].select_dtypes(include=["object","string","category"]).columns.tolist())
    ],
    remainder='drop'
)
class XGBRegressorWrapper(XGBRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def fit(self, X, y, **fit_params):
        X_test_enc = xgb_encoder.transform(X_test[xgb_selected_cols])
        self.eval_set = [(X_test_enc, y_test)]
        return super().fit(X, y,
                              eval_set=self.eval_set,
                              **fit_params,
                              verbose=False)
    def predict(self, X):
        return super().predict(X)


xgb = Pipeline(
    steps=[
        ('enc', xgb_encoder),
        ('model',XGBRegressorWrapper(
            min_child_weight=1,
            n_estimators=1000,
            learning_rate=0.01,
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

#%% get xgboost importance

preprocessor = xgb.named_steps['enc']

feature_names = preprocessor.get_feature_names_out()

xgb_model = xgb.named_steps['model']
importance_df = pd.DataFrame({
    'feature': feature_names,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(100))


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
        ('cat', catboost)
        # ('lgb', lgb)
        # ('linear', linear)
    ],
    final_estimator=XGBRegressor(
        n_estimators=500,
        learning_rate=0.02,
        max_depth=2,
        subsample=0.8,
        colsample_bytree=0.8,
        objective='reg:absoluteerror',
        random_state=SEED,
        n_jobs=-1,
        verbosity=0
    ),
    cv=KFold(n_splits=7, shuffle=True, random_state=42)
)
final.fit(X_train,y_train)
print_errors(final,X_train,y_train,X_test,y_test)

#%%
final.fit(X,y)
do_test(final, train_df)

# %%
