import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

data = pd.read_csv("CPT_II_ConnersContinuousPerformanceTest (1).csv", delimiter=";")
data.columns = data.columns.str.strip()

if "Adhd Confidence Index" in data.columns:
    data.rename(columns={"Adhd Confidence Index": "label"}, inplace=True)

if "label" not in data.columns:
    raise ValueError("Dataset must contain a 'label' column for ADHD confidence index")

data = data.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)
for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="ignore")

X = data.drop(columns=["label"])
y = data["label"]

numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

num_imputer = SimpleImputer(strategy="median")
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

for col in numeric_cols:
    lower, upper = X[col].quantile(0.01), X[col].quantile(0.99)
    X[col] = np.clip(X[col], lower, upper)

for col in numeric_cols:
    if abs(X[col].skew()) > 1:
        X[col] = np.log1p(X[col] - X[col].min() + 1)

top_corr = X[numeric_cols].corrwith(y).abs().sort_values(ascending=False).head(5).index.tolist()
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)
X_poly = poly.fit_transform(X[top_corr])
X_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(top_corr))

X = pd.concat([X.reset_index(drop=True), X_poly.reset_index(drop=True)], axis=1)

dup_cols = X.columns[X.columns.duplicated()].tolist()
if dup_cols:
    print(f"⚠️ Removing duplicate columns: {dup_cols}")
    X = X.loc[:, ~X.columns.duplicated()]

scaler = StandardScaler()
X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)

y_bins = pd.qcut(y, q=10, duplicates='drop')
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y_bins, test_size=0.2, random_state=42
)

print("🚀 Starting hyperparameter tuning (CPU safe)...")

lgbm = lgb.LGBMRegressor(
    device_type='cpu',
    boosting_type='gbdt',
    n_jobs=-1,
    random_state=42
)
lgbm_params = {
    'n_estimators': [300, 600, 900],
    'learning_rate': [0.01, 0.02, 0.05],
    'num_leaves': [31, 50, 70],
    'max_depth': [-1, 10, 15],
    'colsample_bytree': [0.8, 1.0],
    'subsample': [0.8, 1.0]
}
lgbm_search = RandomizedSearchCV(lgbm, lgbm_params, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1)
lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_
print(f"✅ Best LGBM Params: {lgbm_search.best_params_}")

xgbr = xgb.XGBRegressor(
    tree_method='hist',
    n_jobs=-1,
    eval_metric="rmse",
    random_state=42
)
xgb_params = {
    'n_estimators': [300, 600, 900],
    'learning_rate': [0.01, 0.02, 0.05],
    'max_depth': [4, 6, 8],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_search = RandomizedSearchCV(xgbr, xgb_params, n_iter=10, cv=3, random_state=42, n_jobs=-1, verbose=1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"✅ Best XGBoost Params: {xgb_search.best_params_}")

selector = SelectFromModel(best_xgb, threshold="median", prefit=True)
X_train_sel = selector.transform(X_train)
X_test_sel = selector.transform(X_test)

estimators = [
    ("lgb", best_lgbm),
    ("xgb", best_xgb),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)),
    ("cat", CatBoostRegressor(iterations=600, learning_rate=0.02, depth=8, verbose=False, random_seed=42, task_type="CPU")),
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), learning_rate_init=0.001,
                         early_stopping=True, max_iter=700, random_state=42))
]

meta_learner = RidgeCV(alphas=[0.1, 1.0, 10.0])

stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_learner,
    n_jobs=-1,
    cv=3
)

print("\n⚙️ Training final ensemble model... (CPU optimized)")
stacked_model.fit(X_train_sel, y_train)

y_pred = stacked_model.predict(X_test_sel)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n✅ Model Performance:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

joblib.dump(stacked_model, "adhd_cpu_fixed_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(selector, "feature_selector.joblib")
joblib.dump(X.columns.tolist(), "feature_columns.joblib")
print(f"✅ Saved {len(X.columns)} feature columns for explainability.")

print("\n📦 Model, scaler, and feature selector saved successfully!")

