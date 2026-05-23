# ============================================================
#  CodeAlpha Internship — Task 1: Iris Flower Classification
#  Dataset : Iris.csv (downloaded from Kaggle)
#  Place Iris.csv in the same folder as this script
# ============================================================

# ── 1. Imports ───────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)

# ── 2. Load Dataset ──────────────────────────────────────────
df = pd.read_csv("Iris.csv")        # Iris.csv same folder mein hona chahiye

print("✅ Dataset loaded successfully!")
print("Shape  :", df.shape)
print("\nFirst 5 rows:")
print(df.head())
print("\nColumn names:", df.columns.tolist())

# Drop 'Id' column if it exists (Kaggle version has it)
if "Id" in df.columns:
    df.drop(columns=["Id"], inplace=True)

species_col = "Species"

print("\nClass distribution:")
print(df[species_col].value_counts())

# ── 3. EDA — Exploratory Data Analysis ──────────────────────

# 3a. Class distribution bar chart
plt.figure(figsize=(5, 4))
sns.countplot(x=species_col, data=df, palette="Set2")
plt.title("Class Distribution")
plt.xlabel("Species")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig("eda_class_distribution.png", dpi=150)
plt.show()
print("📊 Saved: eda_class_distribution.png")

# 3b. Pairplot
pair = sns.pairplot(df, hue=species_col, palette="Set2", diag_kind="kde")
pair.fig.suptitle("Pairplot of Iris Features", y=1.02)
plt.savefig("eda_pairplot.png", dpi=150)
plt.show()
print("📊 Saved: eda_pairplot.png")

# 3c. Correlation heatmap
feature_cols = [c for c in df.columns if c != species_col]
plt.figure(figsize=(6, 5))
sns.heatmap(df[feature_cols].corr(), annot=True, fmt=".2f",
            cmap="coolwarm", square=True)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("eda_correlation.png", dpi=150)
plt.show()
print("📊 Saved: eda_correlation.png")

# 3d. Boxplots
fig, axes = plt.subplots(2, 2, figsize=(10, 7))
for ax, col in zip(axes.flatten(), feature_cols):
    df.boxplot(column=col, by=species_col, ax=ax,
               boxprops=dict(color="steelblue"),
               medianprops=dict(color="red"))
    ax.set_title(col)
    ax.set_xlabel("")
fig.suptitle("Feature Distribution per Species", fontsize=13)
plt.tight_layout()
plt.savefig("eda_boxplots.png", dpi=150)
plt.show()
print("📊 Saved: eda_boxplots.png")

# ── 4. Preprocessing ─────────────────────────────────────────
X = df[feature_cols]
y_raw = df[species_col]

le = LabelEncoder()
y = le.fit_transform(y_raw)
target_names = le.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

print(f"\nTrain samples : {X_train.shape[0]}")
print(f"Test  samples : {X_test.shape[0]}")

# ── 5. Train 3 Models ────────────────────────────────────────
models = {
    "K-Nearest Neighbors"  : KNeighborsClassifier(n_neighbors=5),
    "Support Vector Machine": SVC(kernel="rbf", C=1, gamma="scale", random_state=42),
    "Decision Tree"        : DecisionTreeClassifier(max_depth=4, random_state=42),
}

results = {}

for name, model in models.items():
    model.fit(X_train_sc, y_train)
    y_pred = model.predict(X_test_sc)

    acc = accuracy_score(y_test, y_pred)
    cv  = cross_val_score(model, scaler.transform(X), y, cv=5, scoring="accuracy")

    results[name] = {
        "model"   : model,
        "y_pred"  : y_pred,
        "accuracy": acc,
        "cv_mean" : cv.mean(),
        "cv_std"  : cv.std(),
    }

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Test Accuracy : {acc*100:.2f}%")
    print(f"  CV Accuracy   : {cv.mean()*100:.2f}% ± {cv.std()*100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

# ── 6. Confusion Matrices ────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
for ax, (name, r) in zip(axes, results.items()):
    cm   = confusion_matrix(y_test, r["y_pred"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                                   display_labels=target_names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"{name}\nAcc: {r['accuracy']*100:.1f}%")
    ax.tick_params(axis="x", rotation=30)
plt.suptitle("Confusion Matrices — All Models", fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("confusion_matrices.png", dpi=150)
plt.show()
print("📊 Saved: confusion_matrices.png")

# ── 7. Model Comparison Chart ────────────────────────────────
names     = list(results.keys())
test_accs = [r["accuracy"] for r in results.values()]
cv_means  = [r["cv_mean"]  for r in results.values()]
cv_stds   = [r["cv_std"]   for r in results.values()]

x     = np.arange(len(names))
width = 0.35

fig, ax = plt.subplots(figsize=(9, 5))
b1 = ax.bar(x - width/2, test_accs, width, label="Test Accuracy",
            color="#4C72B0", alpha=0.85)
b2 = ax.bar(x + width/2, cv_means,  width, label="CV Accuracy (mean)",
            color="#55A868", alpha=0.85, yerr=cv_stds, capsize=5)

ax.set_ylim(0.85, 1.03)
ax.set_ylabel("Accuracy")
ax.set_title("Model Comparison — Test vs Cross-Validation Accuracy")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=10, ha="right")
ax.legend()
ax.bar_label(b1, fmt="%.2f", padding=3, fontsize=9)
ax.bar_label(b2, fmt="%.2f", padding=3, fontsize=9)
plt.tight_layout()
plt.savefig("model_comparison.png", dpi=150)
plt.show()
print("📊 Saved: model_comparison.png")

# ── 8. Best Model & Sample Prediction ───────────────────────
best_name  = max(results, key=lambda k: results[k]["cv_mean"])
best       = results[best_name]
best_model = best["model"]

print("\n" + "="*50)
print(f"  🏆 Best Model : {best_name}")
print(f"  CV Accuracy  : {best['cv_mean']*100:.2f}% ± {best['cv_std']*100:.2f}%")
print("="*50)

sample    = np.array([[5.1, 3.5, 1.4, 0.2]])
sample_sc = scaler.transform(sample)
pred      = best_model.predict(sample_sc)
print(f"\n🔍 Sample input     : {sample[0]}")
print(f"   Predicted species: {target_names[pred[0]]}")

print("\n✅ Task 1 complete! Saare charts PNG mein save ho gaye hain.")
print("\n📁 GitHub par ye files upload karo:")
print("   - task1_iris_classification.py")
print("   - Iris.csv")
print("   - eda_class_distribution.png")
print("   - eda_pairplot.png")
print("   - eda_correlation.png")
print("   - eda_boxplots.png")
print("   - confusion_matrices.png")
print("   - model_comparison.png")
