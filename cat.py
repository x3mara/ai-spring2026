#%%
from util import *

#%%
X,y,train_df = get_train()

for col in train_df.select_dtypes(include=["object","str"]).columns:
    train_df[col] = train_df[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)

cat_selected_cols = [
    'Item_Category', 'Item_Fat_Content',
    'Item_Identifier',
    'Item_Type',
    'Item_Visibility_Log', 
    'Outlet_Location_Tier', 'Outlet_Size',
    'Outlet_Type',
    'Item_MRP', 'MRP_OutletType', 
    'Outlet_Avg_MRP', 'Item_Price_Rank'
]
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
    learning_rate=0.03,
    early_stopping_rounds=100,
    use_best_model=True,
    depth=6,
    l2_leaf_reg=6,
    loss_function='MAE',
    eval_metric='MAE',
    verbose=50
)

catboost.fit(X_train, y_train)
print_errors(catboost, X_train, y_train, X_test, y_test)

# %%
importance_df = pd.DataFrame({
    'feature': catboost.feature_names_,
    'importance': catboost.feature_importances_
}).sort_values('importance', ascending=False)
print(importance_df.head(100))

# %%