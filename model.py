import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import joblib

print("Downloading JM1 Dataset from OpenML (identical to Kaggle version)...")
# 1. IMPORT REAL DATASET
jm1 = fetch_openml(name='jm1', version=1, parser='auto', as_frame=True)
df = jm1.frame

# 2. DATA CLEANING & RENAMING
print("Cleaning data and mapping readable feature names...")
df = df.replace('?', np.nan)
df = df.dropna()

# Dictionary to map cryptic JM1 column names to readable engineering terms
feature_name_mapping = {
    'loc': 'Lines of Code (LOC)',
    'v(g)': 'Cyclomatic Complexity (v(g))',
    'ev(g)': 'Essential Complexity (ev(g))',
    'iv(g)': 'Design Complexity (iv(g))',
    'n': 'Halstead Length (n)',
    'v': 'Halstead Volume (v)',
    'l': 'Halstead Program Length (l)',
    'd': 'Halstead Difficulty (d)',
    'i': 'Halstead Intelligence (i)',
    'e': 'Halstead Effort (e)',
    'b': 'Halstead Delivered Bugs (b)',
    't': 'Halstead Time Estimator (t)',
    'lOCode': 'Lines of Executable Code',
    'lOComment': 'Lines of Comments',
    'lOBlank': 'Lines of Blank Space',
    'locCodeAndComment': 'Lines of Code & Comments',
    'uniq_Op': 'Unique Operators',
    'uniq_Opnd': 'Unique Operands',
    'total_Op': 'Total Operators',
    'total_Opnd': 'Total Operands',
    'branchCount': 'Branch Count'
}

# Apply numeric conversion and rename the columns
X = df.drop('defects', axis=1).apply(pd.to_numeric)
X = X.rename(columns=feature_name_mapping)

y = df['defects'].astype(str).str.lower().map({'true': 1, 'false': 0})

averages = X.mean()

# 3. TRAIN THE MODEL
print("Training the Random Forest Classifier...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)
print(f"\nModel Accuracy on Test Data: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(f"F1 Score on Test Data:       {f1_score(y_test, y_pred):.4f}")
print("-" * 50)

# 4. THE "RISK & REASON" ANALYZER ENGINE
def analyze_code_risk(model, features, metric_averages):
    prob_defective = model.predict_proba(features)[0][1]

    print(f"🚨 DEFECT PROBABILITY: {prob_defective * 100:.1f}%")

    if prob_defective > 0.50:
        print("STATUS: High Risk - Code Review Recommended")
        print("REASONS:")

        importances = model.feature_importances_
        feature_names = features.columns

        importance_dict = dict(zip(feature_names, importances))
        sorted_features = sorted(importance_dict.items(), key=lambda item: item[1], reverse=True)

        reasons_found = 0
        for feature_name, _ in sorted_features:
            actual_value = features[feature_name].iloc[0]
            avg_value = metric_averages[feature_name]

            if actual_value > (avg_value * 1.2):
                print(f"  -> Metric '{feature_name}' is suspiciously high! (Actual: {actual_value:.2f} | Safe Avg: {avg_value:.2f})")
                reasons_found += 1

            if reasons_found >= 3:
                break

        if reasons_found == 0:
            print("  -> Multiple metrics are slightly elevated, causing cumulative risk.")
    else:
        print("STATUS: Low Risk - Code metrics look stable.")

    print("-" * 50)

# 5. GENERATE VISUALIZATIONS
print("\n--- GENERATING VISUALIZATIONS (TEST SET) ---")

test_df = X_test.copy()
test_df['defects'] = y_test

# Plot A: Feature Importance Bar Chart
plt.figure(figsize=(10, 8)) # Slightly taller to fit the longer names
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
top_n = 10
top_indices = indices[:top_n]

sns.barplot(x=importances[top_indices], y=X.columns[top_indices], palette='viridis', hue=X.columns[top_indices], legend=False)
plt.title('Top 10 Learned Feature Importances', fontsize=14)
plt.xlabel('Importance Score (Influence on the model)', fontsize=12)
plt.ylabel('Software Metric', fontsize=12)
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.show()

# Plot B: Distribution Boxplot (Now using the mapped LOC name)
plt.figure(figsize=(8, 6))
# Filter extreme outliers using the new column name
plot_data = test_df[test_df['Lines of Code (LOC)'] < 500]

sns.boxplot(x='defects', y='Lines of Code (LOC)', data=plot_data, palette='Set2', hue='defects', legend=False)
plt.title('Distribution of LOC (Test Data: Clean vs Defective)', fontsize=14)
plt.xlabel('Is Defective?', fontsize=12)
plt.ylabel('Lines of Code (LOC)', fontsize=12)
plt.xticks([0, 1], ['Clean (0)', 'Defective (1)'])
plt.tight_layout()
plt.savefig('metric_distribution_test.png')
plt.show()

print("Test-set graphs displayed and saved to your local directory successfully!\n")

# 6. TEST IT OUT WITH REAL DATA
print("--- RUNNING RISK ANALYZER ON 10 SAMPLE FILES FROM TEST SET ---\n")

safe_indices = y_test[y_test == 0].sample(5, random_state=42).index
buggy_indices = y_test[y_test == 1].sample(5, random_state=42).index

print("=== TESTING 5 KNOWN SAFE FILES (ACTUAL: CLEAN) ===")
for i, idx in enumerate(safe_indices, 1):
    print(f"\n[Safe File #{i} | Index: {idx}]")
    sample_file = X_test.loc[[idx]]
    analyze_code_risk(rf_model, sample_file, averages)

print("\n=== TESTING 5 KNOWN DEFECTIVE FILES (ACTUAL: BUGGY) ===")
for i, idx in enumerate(buggy_indices, 1):
    print(f"\n[Buggy File #{i} | Index: {idx}]")
    sample_file = X_test.loc[[idx]]
    analyze_code_risk(rf_model, sample_file, averages)


joblib.dump(rf_model,'rf_model.pkl')