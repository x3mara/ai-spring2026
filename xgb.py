#%%
from util import *

#%%

X,y,train_df = get_train()

for col in train_df.select_dtypes(include=["object","str"]).columns:
    train_df[col] = train_df[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)


#%% xgb

xgb_selected_cols = [
    'Mean_Y_by_OutletType', 'Item_MRP',
    'Price_Per_Unit_Weight', 'Item_Weight',
    'Item_Visibility_Log'
]

xgb_encoder = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', X[xgb_selected_cols].select_dtypes(include=['int64', 'float64']).columns),
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

if __name__ == "__main__":
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