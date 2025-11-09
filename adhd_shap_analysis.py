import joblib
import shap
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

print("\n🔍 Starting simplified SHAP analysis...")

model = joblib.load("adhd_cpu_fixed_model.joblib")
scaler = joblib.load("scaler.joblib")
feature_selector = joblib.load("feature_selector.joblib")

try:
    feature_columns = joblib.load("feature_columns.joblib")
except FileNotFoundError:
    feature_columns = joblib.load("feature_cols.joblib")
    print("⚠️ Using fallback: feature_cols.joblib")

data = pd.read_csv("CPT_II_ConnersContinuousPerformanceTest (1).csv", delimiter=";")
data.columns = data.columns.str.strip()
data.rename(columns={"Adhd Confidence Index": "label"}, inplace=True)

X = data.drop(columns=["label"])
y = data["label"]

for col in X.columns:
    X[col] = pd.to_numeric(X[col], errors="coerce")
X.fillna(X.median(), inplace=True)

for col in feature_columns:
    if col not in X.columns:
        X[col] = 0
X = X[feature_columns]

X_scaled = scaler.transform(X)
X_selected_array = feature_selector.transform(X_scaled)
selected_mask = feature_selector.get_support()
selected_features = np.array(feature_columns)[selected_mask]

X_selected = pd.DataFrame(X_selected_array, columns=selected_features)
print(f"✅ Data aligned: {X_selected.shape[1]} selected features")

lgbm_model = None
if hasattr(model, "named_estimators_") and "lgb" in model.named_estimators_:
    lgbm_model = model.named_estimators_["lgb"]
else:
    print("⚠️ Using full stacked model for SHAP (slower).")
    lgbm_model = model

X_sample = shap.utils.sample(X_selected, 300, random_state=42)
explainer = shap.TreeExplainer(lgbm_model)
shap_values = explainer(X_sample)

importance_df = pd.DataFrame({
    "Feature": X_selected.columns,
    "Mean_Abs_SHAP": np.abs(shap_values.values).mean(axis=0)
}).sort_values(by="Mean_Abs_SHAP", ascending=False)

importance_df.to_csv("selected_features.csv", index=False)
print("✅ Exported feature importance as 'selected_features.csv'")

top_n = 15
top_features_df = importance_df.head(top_n)

plt.figure(figsize=(10, 6))
plt.barh(top_features_df["Feature"][::-1], top_features_df["Mean_Abs_SHAP"][::-1], color="skyblue")
plt.xlabel("Average |SHAP Value| (Importance)")
plt.ylabel("Feature")
plt.title(f"Top {top_n} Most Important Features")
plt.tight_layout()
plt.savefig("simple_top15_features_bar.png", dpi=300, bbox_inches="tight")
plt.show()
print("✅ Saved: simple_top15_features_bar.png")

try:
    threshold = np.median(y)
    high_mask = y >= threshold
    low_mask = y < threshold

    high_mean = np.abs(shap_values.values[high_mask]).mean(axis=0)
    low_mean = np.abs(shap_values.values[low_mask]).mean(axis=0)

    diff_df = pd.DataFrame({
        "Feature": X_selected.columns,
        "Difference": high_mean - low_mean
    }).sort_values(by="Difference", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    colors = ["salmon" if x > 0 else "lightgreen" for x in diff_df["Difference"][::-1]]
    plt.barh(diff_df["Feature"][::-1], diff_df["Difference"][::-1], color=colors)
    plt.xlabel("SHAP Difference (High ADHD - Low ADHD)")
    plt.title("Top 10 Features Differentiating High vs Low ADHD Predictions")
    plt.tight_layout()
    plt.savefig("simple_high_vs_low_diff_bar.png", dpi=300, bbox_inches="tight")
    plt.show()
    diff_df.to_csv("shap_high_vs_low_diff.csv", index=False)
    print("✅ Saved: simple_high_vs_low_diff_bar.png & shap_high_vs_low_diff.csv")

except Exception as e:
    print(f"⚠️ Skipped high-vs-low comparison: {e}")

print("\n🎉 Simplified SHAP analysis complete!")
print("📊 Generated files:")
print("   • selected_features.csv")
print("   • simple_top15_features_bar.png")
print("   • simple_high_vs_low_diff_bar.png (optional)")
