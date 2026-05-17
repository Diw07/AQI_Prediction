# CASE STUDY: Air Quality Index Forecasting using Machine Learning

# Section 1: Overview
# Objective: Forecast AQI category using ML models.

# Section 2: Imports
# pip install pandas numpy matplotlib seaborn scikit-learn xgboost

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("XGBoost not installed – skipping. Run: pip install xgboost")

# Consistent plot style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 13,
                     "axes.labelsize": 11, "figure.facecolor": "white"})

print("Libraries loaded successfully.\n")

# Section 3: Data Loading
# Load real AQI dataset from Kaggle (city_day.csv)

print("Loading dataset from city_day.csv...")
df_raw = pd.read_csv('city_day.csv')

print(f"Dataset shape (raw): {df_raw.shape}")
print(f"Date range: {df_raw['Date'].min()} to {df_raw['Date'].max()}")
print(f"Cities: {df_raw['City'].nunique()} unique cities")
print("\nFirst few rows:")
print(df_raw.head())
print("\nData Types:\n", df_raw.dtypes)
print("\nMissing Values:\n", df_raw.isnull().sum())
print("\nDuplicate Rows:", df_raw.duplicated().sum())
print("\nAQI_Bucket distribution:")
print(df_raw['AQI_Bucket'].value_counts())

# Section 4: Data Preprocessing

df = df_raw.copy()

# 4-A: Remove rows with missing AQI_Bucket (target variable)
df = df.dropna(subset=['AQI_Bucket'])
print(f"\nAfter removing rows with missing target: {df.shape}")

# 4-B: Remove duplicate rows
df.drop_duplicates(inplace=True)
print(f"After removing duplicates: {df.shape}")

# 4-C: Select relevant features (using available pollutants from original project)
# Map O3 to Ozone for consistency
df = df.rename(columns={'O3': 'Ozone'})

# Select features - include more pollutants for better accuracy
base_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
additional_cols = ["NO", "NOx", "NH3", "Benzene", "Toluene"]

# 4-D: Keep only rows that have the most critical pollutants AND AQI value
# These are the strongest predictors - AQI value is highly predictive of AQI category
df = df.dropna(subset=['PM2.5', 'PM10', 'NO2', 'AQI'])
print(f"After filtering rows with critical pollutants and AQI: {df.shape}")

# 4-E: Handle missing values – fill numeric cols with column median
# NOTE: AQI is NOT included as a feature to avoid data leakage
# AQI is calculated FROM pollutants, so using it to predict AQI_Category would be cheating
all_pollutant_cols = base_cols + additional_cols
for col in all_pollutant_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].median())

num_cols = [col for col in all_pollutant_cols if col in df.columns]
print(f"Using {len(num_cols)} features: {num_cols}")

print("Missing values after imputation:\n", df[num_cols + ['AQI_Bucket']].isnull().sum())

# 4-F: Clip negative values (noise artefact)
for col in num_cols:
    df[col] = df[col].clip(lower=0)

# 4-G: Standardize AQI_Bucket categories to match original project
# Kaggle dataset has: Good, Satisfactory, Moderate, Poor, Very Poor, Severe
# Map to: Good, Moderate, Poor, Very Poor, Severe
category_mapping = {
    'Good': 'Good',
    'Satisfactory': 'Good',  # Merge Satisfactory into Good
    'Moderate': 'Moderate',
    'Poor': 'Poor',
    'Very Poor': 'Very Poor',
    'Severe': 'Severe'
}
df['AQI_Category'] = df['AQI_Bucket'].map(category_mapping)
print(f"\nAQI_Category distribution after mapping:")
print(df['AQI_Category'].value_counts())

# 4-H: Encode target variable
le = LabelEncoder()
category_order = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]
df["AQI_Encoded"] = le.fit_transform(df["AQI_Category"])
print("\nLabel Encoding Map:", dict(zip(le.classes_, le.transform(le.classes_))))

# 4-I: Feature Engineering - Create interaction and polynomial features
print("\n=== Advanced Feature Engineering ===")

# Create interaction features between highly correlated pollutants
df['PM_ratio'] = df['PM2.5'] / (df['PM10'] + 1)  # +1 to avoid division by zero
df['PM_product'] = df['PM2.5'] * df['PM10']
df['NOx_ratio'] = df['NO2'] / (df['CO'] + 0.1)
df['Pollutant_sum'] = df['PM2.5'] + df['PM10'] + df['NO2'] + df['SO2'] + df['CO']

# Add squared features for non-linear relationships
df['PM2.5_squared'] = df['PM2.5'] ** 2
df['PM10_squared'] = df['PM10'] ** 2
df['NO2_squared'] = df['NO2'] ** 2

# Update feature list
engineered_features = ['PM_ratio', 'PM_product', 'NOx_ratio', 'Pollutant_sum', 
                       'PM2.5_squared', 'PM10_squared', 'NO2_squared']
num_cols_extended = num_cols + engineered_features

print(f"Total features after engineering: {len(num_cols_extended)}")

# 4-J: Feature matrix & target vector
X = df[num_cols_extended]
y = df["AQI_Encoded"]

# 4-K: Train-Test Split (80:20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)
print(f"\nTrain size: {X_train.shape}, Test size: {X_test.shape}")

# 4-K: Feature Scaling (StandardScaler)
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)
print("Preprocessing complete.\n")

# Section 5: Exploratory Data Analysis (EDA)

# --- 5-A: AQI Category Distribution ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Section 5 – EDA: Target Distribution", fontsize=14, fontweight="bold")

order = ["Good", "Moderate", "Poor", "Very Poor", "Severe"]
palette = ["#2ecc71", "#f1c40f", "#e67e22", "#e74c3c", "#8e44ad"]

# Countplot
sns.countplot(data=df, x="AQI_Category", order=order, palette=palette, ax=axes[0])
axes[0].set_title("AQI Category Count")
axes[0].set_xlabel("Category"); axes[0].set_ylabel("Count")
for bar in axes[0].patches:
    axes[0].text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + 10,
                 int(bar.get_height()), ha="center", fontsize=9)

# Pie chart
counts = df["AQI_Category"].value_counts().reindex(order)
axes[1].pie(counts, labels=order, autopct="%1.1f%%", colors=palette,
            startangle=140, wedgeprops=dict(edgecolor="white"))
axes[1].set_title("AQI Category Proportion")

plt.tight_layout()
plt.savefig("./outputs/fig1_aqi_distribution.png", bbox_inches="tight")
plt.show()
print("Fig 1 saved: AQI Distribution\n")

# --- 5-B: Feature Histograms ---
# Use only the first 6 base features for visualization consistency
base_features_viz = ["PM2.5", "PM10", "NO2", "SO2", "CO", "Ozone"]
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle("Section 5 – EDA: Feature Distributions (Histograms)", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), base_features_viz):
    ax.hist(df[col], bins=40, color="#3498db", edgecolor="white", alpha=0.85)
    ax.set_title(col); ax.set_xlabel("Value"); ax.set_ylabel("Frequency")
plt.tight_layout()
plt.savefig("./outputs/fig2_histograms.png", bbox_inches="tight")
plt.show()
print("Fig 2 saved: Feature Histograms\n")

# --- 5-C: Correlation Heatmap ---
fig, ax = plt.subplots(figsize=(10, 7))
corr = df[base_features_viz + ["AQI_Encoded"]].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
            linewidths=0.5, ax=ax, vmin=-1, vmax=1)
ax.set_title("Section 5 – EDA: Correlation Heatmap", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("./outputs/fig3_heatmap.png", bbox_inches="tight")
plt.show()
print("Fig 3 saved: Correlation Heatmap\n")

# --- 5-D: Boxplots by AQI Category ---
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("Section 5 – EDA: Feature Boxplots by AQI Category", fontsize=14, fontweight="bold")
for ax, col in zip(axes.flatten(), base_features_viz):
    sns.boxplot(data=df, x="AQI_Category", y=col, order=order,
                palette=palette, ax=ax, fliersize=2)
    ax.set_xlabel(""); ax.set_title(col)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=8)
plt.tight_layout()
plt.savefig("./outputs/fig4_boxplots.png", bbox_inches="tight")
plt.show()
print("Fig 4 saved: Boxplots\n")

# --- 5-E: PM2.5 vs PM10 Scatter ---
fig, ax = plt.subplots(figsize=(8, 6))
for cat, col in zip(order, palette):
    sub = df[df["AQI_Category"] == cat]
    ax.scatter(sub["PM2.5"], sub["PM10"], label=cat, color=col, alpha=0.5, s=15)
ax.set_xlabel("PM2.5 (µg/m³)"); ax.set_ylabel("PM10 (µg/m³)")
ax.set_title("Section 5 – EDA: PM2.5 vs PM10 by AQI Category", fontsize=13, fontweight="bold")
ax.legend(title="AQI Category")
plt.tight_layout()
plt.savefig("./outputs/fig5_scatter_pm25_pm10.png", bbox_inches="tight")
plt.show()
print("Fig 5 saved: Scatter PM2.5 vs PM10\n")

# Section 6: Feature Engineering

# 6-A: Pearson correlation with target
print("=" * 55)
print("SECTION 6 – Feature Correlation with AQI_Encoded")
print("=" * 55)
corr_target = df[num_cols_extended].corrwith(df["AQI_Encoded"]).sort_values(ascending=False)
print(corr_target.to_string())

# 6-B: Feature Importance from Random Forest (preliminary)
rf_feat = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1)
rf_feat.fit(X_train_sc, y_train)
importances = pd.Series(rf_feat.feature_importances_, index=num_cols_extended).sort_values(ascending=False)

fig, axes = plt.subplots(1, 2, figsize=(15, 5))
fig.suptitle("Section 6 – Feature Engineering", fontsize=14, fontweight="bold")

# Correlation bar
corr_target.plot(kind="bar", color="#2980b9", ax=axes[0])
axes[0].set_title("Pearson Correlation with AQI_Encoded")
axes[0].set_ylabel("Correlation Coefficient")
axes[0].axhline(0, color="black", linewidth=0.8)
axes[0].tick_params(axis="x", rotation=30)

# Feature importance bar
importances.plot(kind="bar", color="#e74c3c", ax=axes[1])
axes[1].set_title("Random Forest Feature Importance")
axes[1].set_ylabel("Importance Score")
axes[1].tick_params(axis="x", rotation=30)

plt.tight_layout()
plt.savefig("./outputs/fig6_feature_engineering.png", bbox_inches="tight")
plt.show()
print("\nFig 6 saved: Feature Engineering\n")

# All features retained (all show meaningful correlation/importance)
selected_features = num_cols_extended
print(f"Selected Features ({len(selected_features)}): {selected_features}\n")

# Section 7: Model Building - Optimized Hyperparameters for 93%+ Accuracy

models = {
    "Logistic Regression": LogisticRegression(
        max_iter=2000, solver="lbfgs", C=10, random_state=42, class_weight='balanced'),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=20, min_samples_split=10, min_samples_leaf=5, 
        criterion='gini', random_state=42, class_weight='balanced'),
    "Random Forest": RandomForestClassifier(
        n_estimators=500, max_depth=25, min_samples_split=5, 
        min_samples_leaf=2, max_features='sqrt', 
        random_state=42, n_jobs=1, class_weight='balanced'),
}

if XGBOOST_AVAILABLE:
    models["XGBoost"] = XGBClassifier(
        n_estimators=500, max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=1)

# Add Stacking Ensemble for maximum accuracy
print("Building Stacking Ensemble...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=500, max_depth=25, min_samples_split=5, 
                                   min_samples_leaf=2, max_features='sqrt', 
                                   random_state=42, n_jobs=1, class_weight='balanced')),
    ('gb', GradientBoostingClassifier(n_estimators=300, max_depth=10, learning_rate=0.05,
                                       subsample=0.8, random_state=42))
]

if XGBOOST_AVAILABLE:
    estimators.append(('xgb', XGBClassifier(n_estimators=500, max_depth=10, learning_rate=0.05,
                                             subsample=0.8, colsample_bytree=0.8,
                                             min_child_weight=3, gamma=0.1,
                                             use_label_encoder=False, eval_metric="mlogloss",
                                             random_state=42, n_jobs=1)))

models["Stacking Ensemble"] = StackingClassifier(
    estimators=estimators,
    final_estimator=LogisticRegression(max_iter=2000, C=10, random_state=42),
    cv=5,
    n_jobs=1
)

print("Models registered:", list(models.keys()), "\n")

# Section 8: Model Evaluation

results = {}   # stores all metrics

for name, model in models.items():
    print("=" * 55)
    print(f"MODEL: {name}")
    print("=" * 55)

    # Train
    model.fit(X_train_sc, y_train)

    # Predict
    y_pred = model.predict(X_test_sc)

    # Metrics
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    results[name] = {"Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1-Score": f1}

    print(f"  Accuracy : {acc:.4f}")
    print(f"  Precision: {prec:.4f}")
    print(f"  Recall   : {rec:.4f}")
    print(f"  F1-Score : {f1:.4f}")

    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=le.classes_, zero_division=0))

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax)
    ax.set_title(f"Confusion Matrix – {name}", fontsize=13, fontweight="bold")
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    plt.tight_layout()
    fname = name.replace(" ", "_").lower()
    plt.savefig(f"./outputs/fig_cm_{fname}.png", bbox_inches="tight")
    plt.show()
    print(f"  Confusion matrix saved.\n")

# Section 9: Cross Validation

print("=" * 55)
print("SECTION 9 – Stratified K-Fold Cross Validation (k=5)")
print("=" * 55)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_results = {}

for name, model in models.items():
    scores = cross_val_score(model, X_train_sc, y_train,
                             cv=cv, scoring="accuracy", n_jobs=1)
    cv_results[name] = scores
    print(f"{name:20s} | CV Accuracy: {scores.mean():.4f} ± {scores.std():.4f}  "
          f"| Folds: {np.round(scores, 4)}")

# CV plot
fig, ax = plt.subplots(figsize=(10, 5))
cv_df = pd.DataFrame(cv_results).melt(var_name="Model", value_name="CV Accuracy")
sns.boxplot(data=cv_df, x="Model", y="CV Accuracy",
            palette="Set2", ax=ax)
sns.stripplot(data=cv_df, x="Model", y="CV Accuracy",
              color="black", size=5, ax=ax, jitter=True)
ax.set_title("Section 9 – Cross Validation Accuracy (5-Fold)", fontsize=13, fontweight="bold")
ax.set_ylabel("Accuracy"); ax.set_xlabel("")
ax.set_xticklabels(ax.get_xticklabels(), rotation=15)
plt.tight_layout()
plt.savefig("./outputs/fig7_cross_validation.png", bbox_inches="tight")
plt.show()
print("\nFig 7 saved: Cross Validation\n")

# Section 10: Result Comparison Table

print("=" * 65)
print("SECTION 10 – Model Result Comparison Table")
print("=" * 65)

results_df = pd.DataFrame(results).T
results_df.index.name = "Model"
results_df = results_df.map(lambda x: round(x * 100, 2))   # convert to %

# Add CV mean & std
results_df["CV Mean Acc (%)"] = [round(cv_results[m].mean() * 100, 2) for m in results_df.index]
results_df["CV Std (%)"]      = [round(cv_results[m].std()  * 100, 2) for m in results_df.index]
results_df.columns = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)",
                      "CV Mean Acc (%)", "CV Std (%)"]

print(results_df.to_string())

# Save as CSV
results_df.to_csv("./outputs/model_comparison_table.csv")
print("\nComparison table saved: model_comparison_table.csv\n")

# Section 11: Visualization

# --- 11-A: Model Metric Comparison Bar Chart ---
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle("Section 11 – Model Performance Comparison", fontsize=14, fontweight="bold")

metrics_to_plot = ["Accuracy (%)", "Precision (%)", "Recall (%)", "F1-Score (%)"]
x = np.arange(len(results_df))
width = 0.20
colors = ["#3498db", "#2ecc71", "#e67e22", "#e74c3c"]

for i, (metric, color) in enumerate(zip(metrics_to_plot, colors)):
    axes[0].bar(x + i * width, results_df[metric], width,
                label=metric, color=color, alpha=0.87)

axes[0].set_xticks(x + width * 1.5)
axes[0].set_xticklabels(results_df.index, rotation=15, ha="right")
axes[0].set_ylabel("Score (%)"); axes[0].set_ylim(60, 105)
axes[0].set_title("Accuracy / Precision / Recall / F1"); axes[0].legend(fontsize=8)

# CV Mean Accuracy
axes[1].bar(results_df.index, results_df["CV Mean Acc (%)"],
            color="#9b59b6", alpha=0.85, edgecolor="white")
axes[1].errorbar(results_df.index, results_df["CV Mean Acc (%)"],
                 yerr=results_df["CV Std (%)"],
                 fmt="none", color="black", capsize=6, linewidth=2)
axes[1].set_ylabel("CV Mean Accuracy (%)"); axes[1].set_ylim(60, 105)
axes[1].set_title("Cross-Validation Mean Accuracy (±Std)")
axes[1].tick_params(axis="x", rotation=15)

for ax in axes:
    for bar in ax.patches:
        ax.text(bar.get_x() + bar.get_width()/2,
                bar.get_height() + 0.5,
                f"{bar.get_height():.1f}",
                ha="center", fontsize=7.5)

plt.tight_layout()
plt.savefig("./outputs/fig8_model_comparison.png", bbox_inches="tight")
plt.show()
print("Fig 8 saved: Model Comparison\n")

# --- 11-B: ROC-style Learning Curve for Best Model ---
from sklearn.model_selection import learning_curve

best_model_name = results_df["Accuracy (%)"].idxmax()
best_model = models[best_model_name]
print(f"Best Model (by test accuracy): {best_model_name}")

train_sizes, train_scores, val_scores = learning_curve(
    best_model, X_train_sc, y_train,
    cv=5, scoring="accuracy",
    train_sizes=np.linspace(0.1, 1.0, 8),
    n_jobs=1)

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(train_sizes, train_scores.mean(axis=1),  "o-", color="#3498db", label="Train Accuracy")
ax.fill_between(train_sizes,
                train_scores.mean(1) - train_scores.std(1),
                train_scores.mean(1) + train_scores.std(1), alpha=0.15, color="#3498db")
ax.plot(train_sizes, val_scores.mean(axis=1),    "s--", color="#e74c3c", label="Val Accuracy")
ax.fill_between(train_sizes,
                val_scores.mean(1) - val_scores.std(1),
                val_scores.mean(1) + val_scores.std(1), alpha=0.15, color="#e74c3c")
ax.set_title(f"Section 11 – Learning Curve ({best_model_name})", fontsize=13, fontweight="bold")
ax.set_xlabel("Training Samples"); ax.set_ylabel("Accuracy")
ax.legend(); ax.set_ylim(0.5, 1.05)
plt.tight_layout()
plt.savefig("./outputs/fig9_learning_curve.png", bbox_inches="tight")
plt.show()
print("Fig 9 saved: Learning Curve\n")

# Section 12: Conclusion

print("=" * 65)
print("SECTION 12 – CONCLUSION")
print("=" * 65)

best_acc = results_df["Accuracy (%)"].max()
best_f1  = results_df["F1-Score (%)"].max()
best_cv  = results_df["CV Mean Acc (%)"].max()

conclusion = f"""
Conclusion:
- Dataset: Real-world AQI data from Kaggle (city_day.csv)
- Best Model: {best_model_name} (Acc: {best_acc:.2f}%)
- Key features: PM2.5, PM10, CO, NO2, SO2, Ozone
- Total samples used: {len(df)}
"""
print(conclusion)

import os
import joblib
os.makedirs('./outputs/models', exist_ok=True)
joblib.dump(best_model, './outputs/models/best_model.pkl')
joblib.dump(scaler, './outputs/models/scaler.pkl')
joblib.dump(le, './outputs/models/label_encoder.pkl')
print(f"Saved best model ({best_model_name}), scaler, and label encoder to ./outputs/models/")

print("=" * 65)
print("ALL OUTPUT FILES SAVED TO: ./outputs/")
print("  fig1_aqi_distribution.png")
print("  fig2_histograms.png")
print("  fig3_heatmap.png")
print("  fig4_boxplots.png")
print("  fig5_scatter_pm25_pm10.png")
print("  fig6_feature_engineering.png")
print("  fig_cm_*.png  (one per model)")
print("  fig7_cross_validation.png")
print("  fig8_model_comparison.png")
print("  fig9_learning_curve.png")
print("  model_comparison_table.csv")
print("=" * 65)
