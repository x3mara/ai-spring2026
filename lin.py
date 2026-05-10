#%%
from util import *
from sklearn.preprocessing import PolynomialFeatures

#%%

X,y,train_df = get_train()

for col in train_df.select_dtypes(include=["object","str"]).columns:
    train_df[col] = train_df[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)

#%%

selected_cols = [
    'Item_MRP', 'Price_Per_Unit_Weight',
    'Item_Weight', 'Outlet_Size',
    'Item_Type', 'Item_Category',
    'Outlet_Type_Avg_MRP',
    'Outlet_Avg_MRP'
]

encoder = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), X[selected_cols].select_dtypes(include=['int64', 'float64']).columns),
        ('cat', OneHotEncoder(handle_unknown='ignore'),X[selected_cols].select_dtypes(include=["object","string","category"]).columns.tolist())
    ],
    remainder='drop'
)

lgb = Pipeline(
    steps=[
        ('enc',encoder),
        ('poly',PolynomialFeatures(degree=2)),
        ('model',LinearRegression())
    ]
)

if __name__ == "__main__":
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