"""
train.py
---------------------------------
Train a Random Forest model to predict lane change decisions.

Input features:
- speed
- steering_angle
- dist_left
- dist_right
- road_curve

Output label:
- KEEP_LANE / CHANGE_LEFT / CHANGE_RIGHT
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 60)
print("🚗 LANE CHANGE PREDICTION - MODEL TRAINING")
print("=" * 60)

# ------------------ LOAD DATA ------------------
print("\n📁 Loading dataset (lane_data.csv)...")
df = pd.read_csv("lane_data.csv")
print(f"✅ Dataset loaded: {df.shape[0]} samples, {df.shape[1]} columns")

# X = features, y = target
X = df[["speed", "steering_angle", "dist_left", "dist_right", "road_curve"]]
y = df["action"]

# ------------------ TRAIN / TEST SPLIT ------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n📊 Data split summary:")
print(f"- Training samples: {X_train.shape[0]}")
print(f"- Testing samples : {X_test.shape[0]}")

# ------------------ MODEL TRAINING ------------------
print("\n🤖 Training Random Forest model...")
model = RandomForestClassifier(
    n_estimators=150,
    max_depth=10,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

# ------------------ EVALUATION ------------------
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("\n✅ Model trained successfully!")
print(f"🎯 Test Accuracy: {accuracy:.2%}")

print("\n📈 Detailed Performance (per class):")
print(classification_report(y_test, y_pred))

# ------------------ FEATURE IMPORTANCE ------------------
importance = pd.DataFrame({
    "feature": X.columns,
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False)

print("\n🔍 Feature Importance (higher = more influence on decision):")
print(importance.to_string(index=False))

# Save model for Streamlit app
joblib.dump(model, "lane_model.pkl")
print("\n💾 Trained model saved as 'lane_model.pkl'")

# ------------------ VISUALIZE FEATURE IMPORTANCE ------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=importance, x="importance", y="feature")
plt.title("Feature Importance in Lane Change Prediction")
plt.xlabel("Importance Score")
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=100, bbox_inches="tight")

print("📊 Visualization saved as 'feature_importance.png'")
print("\n" + "=" * 60)
print("🎉 TRAINING COMPLETE!")
print("=" * 60)
