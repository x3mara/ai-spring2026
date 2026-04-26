import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import (
    RandomForestRegressor,
    StackingRegressor,
    VotingRegressor
)
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
import lightgbm as lgb
import os

SEED = 442004


def preprocess(df, train_ref=None):
    # fat content has duplicates written differently
    df['Item_Fat_Content'] = df['Item_Fat_Content'].replace({
        'low fat': 'Low Fat',
        'LF': 'Low Fat',
        'reg': 'Regular'
    })

    # visibility = 0 doesnt make sense so replace with median of that item type
    if train_ref is not None:
        vis_medians = train_ref.groupby('Item_Type')['Item_Visibility'].median()
    else:
        vis_medians = df.groupby('Item_Type')['Item_Visibility'].median()

    df['Item_Visibility'] = df.apply(
        lambda row: vis_medians[row['Item_Type']]
        if row['Item_Visibility'] == 0
        else row['Item_Visibility'],
        axis=1
    )

    # fill item weight using same product's weight from other rows
    if train_ref is not None:
        weight_by_id = train_ref.groupby('Item_Identifier')['Item_Weight'].median()
    else:
        weight_by_id = df.groupby('Item_Identifier')['Item_Weight'].median()

    df['Item_Weight'] = df.apply(
        lambda row: weight_by_id.get(row['Item_Identifier'], np.nan)
        if pd.isna(row['Item_Weight'])
        else row['Item_Weight'],
        axis=1
    )
    df['Item_Weight'] = df['Item_Weight'].fillna(df['Item_Weight'].median())

    # fill all outlet size nulls
    df['Outlet_Size'] = df['Outlet_Size'].fillna('Missing')

    # drop identifiers, theyre just IDs
    df = df.drop(columns=['Item_Identifier', 'Outlet_Identifier'])

    return df


def add_features(df, train_ref_processed=None):
    # outlet age is more useful than the raw year
    df['Outlet_Age'] = 2026 - df['Outlet_Est_Year']

    # log transform visibility since its skewed
    df['Visibility_Log'] = np.log1p(df['Item_Visibility'])

    # group items into price bins
    df['MRP_Bin'] = pd.cut(
        df['Item_MRP'],
        bins=[0, 70, 130, 200, 300],
        labels=[1, 2, 3, 4]
    ).astype(float)

    # price to weight ratio
    df['Price_Per_Weight'] = df['Item_MRP'] / (df['Item_Weight'] + 0.01)

    # is the item perishable (food)
    perishable = ['Dairy', 'Meat', 'Fruits and Vegetables',
                  'Seafood', 'Breads', 'Breakfast']
    df['Is_Perishable'] = df['Item_Type'].apply(
        lambda x: 1 if x in perishable else 0
    )

    # target encoding - average Y per item type and outlet type
    if train_ref_processed is not None and 'Y' not in df.columns:
        type_means = train_ref_processed.groupby('Item_Type')['Y'].mean()
        df['Mean_Y_by_ItemType'] = df['Item_Type'].map(type_means)
        df['Mean_Y_by_ItemType'] = df['Mean_Y_by_ItemType'].fillna(
            train_ref_processed['Y'].mean()
        )

        outlet_means = train_ref_processed.groupby('Outlet_Type')['Y'].mean()
        df['Mean_Y_by_OutletType'] = df['Outlet_Type'].map(outlet_means)
        df['Mean_Y_by_OutletType'] = df['Mean_Y_by_OutletType'].fillna(
            train_ref_processed['Y'].mean()
        )
    else:
        df['Mean_Y_by_ItemType'] = df.groupby('Item_Type')['Y'].transform('mean')
        df['Mean_Y_by_OutletType'] = df.groupby('Outlet_Type')['Y'].transform('mean')

    # ordinal encoding for ordered categories
    size_map = {'Small': 0, 'Medium': 1, 'High': 2, 'Missing': 3}
    df['Outlet_Size_Ordinal'] = df['Outlet_Size'].map(size_map)

    tier_map = {'Tier 1': 0, 'Tier 2': 1, 'Tier 3': 2}
    df['Outlet_Location_Tier_Ordinal'] = df['Outlet_Location_Tier'].map(tier_map)

    return df


def handle_outliers(X, y, method='cap'):
    Q1 = y.quantile(0.25)
    Q3 = y.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    outliers = ((y < lower) | (y > upper)).sum()
    print(f"found {outliers} outliers out of {len(y)} rows")

    if method == 'cap':
        y_clean = y.clip(lower=lower, upper=upper)
        return X.copy(), y_clean
    elif method == 'remove':
        mask = (y >= lower) & (y <= upper)
        return X[mask].reset_index(drop=True), y[mask].reset_index(drop=True)


def get_train():
    df = pd.read_csv("train.csv")

    global train_raw
    train_raw = df.copy()

    df = preprocess(df)

    global train_processed
    train_processed = df.copy()

    df = add_features(df)

    columns = set(df.columns)
    X = df[sorted(list(columns - {'Y'}))]
    y = df['Y']
    return X, y


def get_test():
    df = pd.read_csv("test.csv")
    df = preprocess(df, train_ref=train_raw)
    df = add_features(df, train_ref_processed=train_processed)
    return df[sorted(df.columns)]


def print_errors(name, model, X_tr, y_tr, X_te, y_te):
    pred_tr = model.predict(X_tr)
    pred_te = model.predict(X_te)
    train_mae = mean_absolute_error(y_tr, pred_tr)
    test_mae = mean_absolute_error(y_te, pred_te)

    print(f"\n--- {name} ---")
    print(f"  train MAE: {train_mae:.4f}")
    print(f"  test MAE:  {test_mae:.4f}")
    if test_mae - train_mae > train_mae * 0.5:
        print(f"  (might be overfitting)")

    return pred_te


# load data
X, y = get_train()
X, y = handle_outliers(X, y, method='cap')

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)

print(f"train: {X_train.shape[0]} rows, test: {X_test.shape[0]} rows")
print(f"features: {X_train.shape[1]}")

# encode categorical columns for models that cant handle strings
cat_columns = X_train.select_dtypes(include='object').columns.tolist()
print(f"categorical columns: {cat_columns}")

X_train_enc = pd.get_dummies(X_train, drop_first=True)
X_test_enc = pd.get_dummies(X_test, drop_first=True)
X_train_enc, X_test_enc = X_train_enc.align(X_test_enc, join='left', axis=1, fill_value=0)

results = {}

# model 1 - linear regression
lin_reg = LinearRegression()
lin_reg.fit(X_train_enc, y_train)

lin_pred = print_errors("Linear Regression", lin_reg,
                        X_train_enc, y_train, X_test_enc, y_test)

results['Linear Regression'] = {
    'model': lin_reg, 'predictions': lin_pred,
    'mae': mean_absolute_error(y_test, lin_pred), 'encoded': True
}

# model 2 - xgboost
xgb_reg = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=3,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    early_stopping_rounds=20
)

xgb_reg.fit(X_train_enc, y_train,
            eval_set=[(X_test_enc, y_test)],
            verbose=False)

xgb_pred = print_errors("XGBoost", xgb_reg,
                        X_train_enc, y_train, X_test_enc, y_test)

results['XGBoost'] = {
    'model': xgb_reg, 'predictions': xgb_pred,
    'mae': mean_absolute_error(y_test, xgb_pred), 'encoded': True
}

# model 3 - random forest
rf_reg = RandomForestRegressor(
    n_estimators=900,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=3,
    random_state=SEED,
    n_jobs=-1
)

rf_reg.fit(X_train_enc, y_train)

rf_pred = print_errors("Random Forest", rf_reg,
                       X_train_enc, y_train, X_test_enc, y_test)

results['Random Forest'] = {
    'model': rf_reg, 'predictions': rf_pred,
    'mae': mean_absolute_error(y_test, rf_pred), 'encoded': True
}

# model 4 - catboost (uses original data, handles categories on its own)
cat_reg = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    loss_function='MAE',
    early_stopping_rounds=20,
    random_state=SEED
)

cat_features = X_train.select_dtypes(include='object').columns.tolist()

cat_reg.fit(X_train, y_train,
            cat_features=cat_features,
            eval_set=(X_test, y_test),
            verbose=False)

cat_pred = print_errors("CatBoost", cat_reg,
                        X_train, y_train, X_test, y_test)

results['CatBoost'] = {
    'model': cat_reg, 'predictions': cat_pred,
    'mae': mean_absolute_error(y_test, cat_pred), 'encoded': False
}

# model 5 - lightgbm (was imported in old code but never used)
lgb_reg = lgb.LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=SEED,
    verbose=-1
)

lgb_reg.fit(X_train_enc, y_train,
            eval_set=[(X_test_enc, y_test)],
            callbacks=[lgb.early_stopping(20), lgb.log_evaluation(0)])

lgb_pred = print_errors("LightGBM", lgb_reg,
                        X_train_enc, y_train, X_test_enc, y_test)

results['LightGBM'] = {
    'model': lgb_reg, 'predictions': lgb_pred,
    'mae': mean_absolute_error(y_test, lgb_pred), 'encoded': True
}

# stacking ensemble
print("\ntraining stacking ensemble (this takes a bit)...")
stacker = StackingRegressor(
    estimators=[
        ('xgb', XGBRegressor(n_estimators=500, learning_rate=0.05,
                              max_depth=3, subsample=0.8, random_state=SEED)),
        ('rf', RandomForestRegressor(n_estimators=500, max_depth=15,
                                     random_state=SEED, n_jobs=-1)),
        ('lgb', lgb.LGBMRegressor(n_estimators=500, learning_rate=0.05,
                                   max_depth=6, num_leaves=31,
                                   random_state=SEED, verbose=-1))
    ],
    final_estimator=Ridge(alpha=1.0),
    cv=5,
    n_jobs=-1
)

stacker.fit(X_train_enc, y_train)

stack_pred = print_errors("Stacking", stacker,
                          X_train_enc, y_train, X_test_enc, y_test)

results['Stacking'] = {
    'model': stacker, 'predictions': stack_pred,
    'mae': mean_absolute_error(y_test, stack_pred), 'encoded': True
}

# compare all models
print("\n\n========= MODEL COMPARISON =========")
sorted_results = sorted(results.items(), key=lambda x: x[1]['mae'])
for name, data in sorted_results:
    print(f"  {name:<25} MAE: {data['mae']:.4f}")

best_name = sorted_results[0][0]
best_data = sorted_results[0][1]
print(f"\nbest model: {best_name}")

# generate submission
test_data = get_test()

if best_data['encoded']:
    test_enc = pd.get_dummies(test_data, drop_first=True)
    test_enc = test_enc.reindex(columns=X_train_enc.columns, fill_value=0)
    prediction = best_data['model'].predict(test_enc)
else:
    prediction = best_data['model'].predict(test_data)

submission = pd.DataFrame({
    'row_id': range(len(prediction)),
    'Y': prediction
})
submission.to_csv('output.csv', index=False)
print(f"saved output.csv ({len(prediction)} rows)")


# ============ PLOTS ============

os.makedirs('plots', exist_ok=True)

# target distribution
plt.figure(figsize=(10, 5))
plt.hist(y, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
plt.axvline(y.mean(), color='red', linestyle='--', label=f'Mean = {y.mean():.2f}')
plt.axvline(y.median(), color='green', linestyle='--', label=f'Median = {y.median():.2f}')
plt.title('Distribution of Target Variable (Y)')
plt.xlabel('Y')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plots/01_target_distribution.png', dpi=150)
plt.close()
print("saved plots/01_target_distribution.png")

# correlation heatmap
plt.figure(figsize=(14, 10))
numeric_data = pd.concat([X_train_enc, y_train], axis=1)
top_corr = numeric_data.corr()['Y'].abs().sort_values(ascending=False).head(16).index
sns.heatmap(numeric_data[top_corr].corr(), annot=True, fmt='.2f',
            cmap='coolwarm', center=0, square=True, linewidths=0.5)
plt.title('Correlation Heatmap (Top 15 Features)')
plt.tight_layout()
plt.savefig('plots/02_correlation_heatmap.png', dpi=150)
plt.close()
print("saved plots/02_correlation_heatmap.png")

# feature importance
importances = xgb_reg.feature_importances_
imp_df = pd.DataFrame({
    'Feature': X_train_enc.columns,
    'Importance': importances
}).sort_values('Importance', ascending=True).tail(15)

plt.figure(figsize=(10, 8))
plt.barh(imp_df['Feature'], imp_df['Importance'], color='steelblue')
plt.title('XGBoost Feature Importance (Top 15)')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('plots/03_feature_importance.png', dpi=150)
plt.close()
print("saved plots/03_feature_importance.png")

# model comparison
sorted_models = sorted(results.items(), key=lambda x: x[1]['mae'])
names = [n for n, _ in sorted_models]
maes = [d['mae'] for _, d in sorted_models]

plt.figure(figsize=(12, 6))
colors = ['#e74c3c', '#e67e22', '#f1c40f', '#2ecc71', '#3498db', '#9b59b6']
bars = plt.bar(names, maes, color=colors[:len(names)], edgecolor='black')
for bar, score in zip(bars, maes):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
             f'{score:.4f}', ha='center', fontweight='bold')
plt.title('Model Comparison (MAE - Lower is Better)')
plt.ylabel('MAE')
plt.ylim(0, max(maes) * 1.15)
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plots/04_model_comparison.png', dpi=150)
plt.close()
print("saved plots/04_model_comparison.png")

# actual vs predicted
plt.figure(figsize=(8, 8))
plt.scatter(y_test, best_data['predictions'], alpha=0.3, s=20, color='steelblue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()],
         'r--', linewidth=2, label='Perfect Prediction')
plt.xlabel('Actual Y')
plt.ylabel('Predicted Y')
plt.title(f'Actual vs Predicted ({best_name})')
plt.legend()
plt.tight_layout()
plt.savefig('plots/05_actual_vs_predicted.png', dpi=150)
plt.close()
print("saved plots/05_actual_vs_predicted.png")

# residuals
residuals = y_test.values - best_data['predictions']

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=50, edgecolor='black', color='steelblue', alpha=0.7)
plt.axvline(0, color='red', linestyle='--', label='Zero Error')
plt.axvline(np.mean(residuals), color='orange', linestyle='--',
            label=f'Mean Error = {np.mean(residuals):.4f}')
plt.title(f'Residual Distribution ({best_name})')
plt.xlabel('Actual - Predicted')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('plots/06_residuals.png', dpi=150)
plt.close()
print("saved plots/06_residuals.png")

# Y by outlet type
train_full = pd.read_csv("train.csv")

plt.figure(figsize=(10, 5))
train_full.boxplot(column='Y', by='Outlet_Type')
plt.title('Y by Outlet Type')
plt.suptitle('')
plt.xlabel('Outlet Type')
plt.ylabel('Y')
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig('plots/07_y_by_outlet_type.png', dpi=150)
plt.close()
print("saved plots/07_y_by_outlet_type.png")

# mrp vs y
plt.figure(figsize=(10, 6))
plt.scatter(train_full['Item_MRP'], train_full['Y'], alpha=0.2, s=10, color='steelblue')
plt.title('Item MRP vs Y')
plt.xlabel('Item MRP')
plt.ylabel('Y')
plt.tight_layout()
plt.savefig('plots/08_mrp_vs_y.png', dpi=150)
plt.close()
print("saved plots/08_mrp_vs_y.png")

print("\nall plots saved to /plots")