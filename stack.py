#%%
from util import *
from xgb import xgb
from cat import catboost
from lgb import lgb

#%%
X,y,train_df = get_train()

for col in train_df.select_dtypes(include=["object","str"]).columns:
    train_df[col] = train_df[col].astype("category")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size = 0.2, random_state = SEED)

final = StackingRegressor(
    estimators=[
        ('xgb', xgb),
        ('cat', catboost),
        ('lgb', lgb)
        # ('linear', linear)
    ],
    final_estimator=LinearRegression(),
    cv=KFold(n_splits=7, shuffle=True, random_state=42)
)
final.fit(X_train,y_train)
print_errors(final,X_train,y_train,X_test,y_test)