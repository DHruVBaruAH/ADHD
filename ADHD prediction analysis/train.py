# ==============================================================
# ADHD Confidence Index - Optimized & Tuned Model Training Pipeline
# ==============================================================

# 1. Import dependencies
import pandas as pd
import numpy as np
import joblib
import shap
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor, StackingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
# MODIFIED: Added RandomizedSearchCV for tuning
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ==============================================================
# 2. Load and clean data (No changes in this section)
# ==============================================================
data = pd.read_csv("CPT_II_ConnersContinuousPerformanceTest (1).csv", delimiter=";")
data.columns = data.columns.str.strip()

if "Adhd Confidence Index" in data.columns:
    data.rename(columns={"Adhd Confidence Index": "label"}, inplace=True)

if "label" not in data.columns:
    raise ValueError("Dataset must contain a 'label' column for ADHD confidence index")

data = data.applymap(lambda x: str(x).replace(",", ".") if isinstance(x, str) else x)

for col in data.columns:
    data[col] = pd.to_numeric(data[col], errors="ignore")

# ==============================================================
# 3. Separate features and target (No changes in this section)
# ==============================================================
X = data.drop(columns=["label"])
y = data["label"]
numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

# ==============================================================
# 4. Preprocessing (No changes in this section)
# ==============================================================
num_imputer = SimpleImputer(strategy="median")
X[numeric_cols] = num_imputer.fit_transform(X[numeric_cols])

if categorical_cols:
    cat_imputer = SimpleImputer(strategy="most_frequent")
    X[categorical_cols] = cat_imputer.fit_transform(X[categorical_cols])
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

scaler = StandardScaler()
X[numeric_cols] = scaler.fit_transform(X[numeric_cols])

# ==============================================================
# 5. Train-test split (No changes in this section)
# ==============================================================
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==============================================================
# 6. NEW: Hyperparameter Tuning for Base Models
# ==============================================================
print("🚀 Starting hyperparameter tuning for base models...")

# --- LightGBM Tuning ---
lgbm = lgb.LGBMRegressor(random_state=42)
lgbm_params = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'num_leaves': [20, 31, 40],
    'max_depth': [-1, 10, 20],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
lgbm_search = RandomizedSearchCV(lgbm, lgbm_params, n_iter=25, cv=5, random_state=42, n_jobs=1, verbose=1)
lgbm_search.fit(X_train, y_train)
best_lgbm = lgbm_search.best_estimator_
print(f"Best LGBM Params: {lgbm_search.best_params_}")

# --- XGBoost Tuning ---
xgbr = xgb.XGBRegressor(random_state=42, eval_metric="rmse")
xgb_params = {
    'n_estimators': [200, 400, 600],
    'learning_rate': [0.01, 0.05, 0.1],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
xgb_search = RandomizedSearchCV(xgbr, xgb_params, n_iter=25, cv=5, random_state=42, n_jobs=1, verbose=1)
xgb_search.fit(X_train, y_train)
best_xgb = xgb_search.best_estimator_
print(f"Best XGBoost Params: {xgb_search.best_params_}")


# ==============================================================
# 7. MODIFIED: Define base models with tuned parameters
# ==============================================================
# Use the best models found during the search
estimators = [
    ("lgb", best_lgbm),
    ("xgb", best_xgb),
    ("rf", RandomForestRegressor(n_estimators=300, random_state=42)), # Can also be tuned
    ("mlp", MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42)) # Can also be tuned
]

# Meta learner
meta_learner = LinearRegression()

# MODIFIED: Stacking Regressor with cross-validation
stacked_model = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_learner,
    n_jobs=-1,
    cv=5  # Add 5-fold cross-validation for robust training
)

# ==============================================================
# 8. Train model
# ==============================================================
print("\nTraining final ensemble model... This may take a few minutes.")
stacked_model.fit(X_train, y_train)

# ==============================================================
# 9. Evaluate
# ==============================================================
y_pred = stacked_model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\nModel Performance on Test Set:")
print(f"R² Score: {r2:.4f}")
print(f"RMSE: {rmse:.4f}")

# ==============================================================
# 10. Save artifacts
# ==============================================================
joblib.dump(stacked_model, "adhd_tuned_model.joblib")
joblib.dump(scaler, "scaler.joblib")
joblib.dump(numeric_cols, "feature_cols.joblib")

print("\n✅ Tuned model, scaler, and feature columns saved successfully!")

# ==============================================================
# 11. SHAP Explainability
# ==============================================================
print("\nGenerating SHAP explanations...")
try:
    # We can still explain one of the powerful base models to get feature insights
    explainer = shap.Explainer(stacked_model.named_estimators_["lgb"])
    shap_values = explainer(X_test)

    plt.title("Top Feature Importances (SHAP Summary)")
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig("adhd_tuned_shap_summary.png", bbox_inches="tight")
    print("✅ SHAP summary plot saved as 'adhd_tuned_shap_summary.png'")
except Exception as e:
    print(f"⚠️ SHAP explanation skipped due to: {e}")