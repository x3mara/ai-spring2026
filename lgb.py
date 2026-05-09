#%%
from util import *

#%%

X,y,train_df = get_train()

for col in train_df.select_dtypes(include=["object","str"]).columns:
    train_df[col] = train_df[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)

#%%

lgbm_selected_cols = [
    'Item_MRP', 'Mean_Y_by_OutletType',
    'Price_Per_Unit_Weight', 'Item_Weight'
]

lgbm_encoder = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', X[lgbm_selected_cols].select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'),X[lgbm_selected_cols].select_dtypes(include=["object","string","category"]).columns.tolist())
    ],
    remainder='drop'
)
class LGBMRegressorWrapper(LGBMRegressor):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
    def fit(self, X, y, **fit_params):
        X_test_enc = lgbm_encoder.transform(X_test[lgbm_selected_cols])
        self.eval_set = [(X_test_enc, y_test)]
        return super().fit(X, y,
                              eval_set=self.eval_set,
                              categorical_feature='auto',
                              callbacks=[lgb.log_evaluation(0)],
                              **fit_params)
    def predict(self, X):
        return super().predict(X)

lgb = Pipeline(
    steps=[
        ('enc',lgbm_encoder),
        ('model',LGBMRegressor(
            n_estimators=200,
            learning_rate=0.025,
            max_depth=2,
            # num_leaves=63,
            objective='mae',
            random_state=42,
            verbose=-1
        ))
    ]
)

if __name__ == "main":
    lgb.fit(X_train, y_train)
    print_errors(lgb, X_train, y_train, X_test, y_test)

#%%
    preprocessor = lgb.named_steps['enc']
    feature_names = preprocessor.get_feature_names_out()
    lgb_model = lgb.named_steps['model']
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': lgb_model.feature_importances_
    }).sort_values('importance', ascending=False)
    print(importance_df.head(10))