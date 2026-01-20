import pandas as pd
import joblib
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset
print("Loading Breast Cancer Wisconsin dataset...")
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

print(f"Dataset shape: {df.shape}")
print(f"Missing values: {df.isnull().sum().sum()}")

# 2. Select features
selected_features = [
    'mean radius', 'mean texture', 'mean perimeter',
    'mean area', 'mean smoothness', 'mean compactness'
]
X = df[selected_features]
y = df['target']

# 3. Split data
print("Splitting data (80-20 train-test)...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. Create pipeline with scaling and model
print("\nCreating pipeline with StandardScaler + LogisticRegression...")
pipeline = make_pipeline(
    StandardScaler(),
    LogisticRegression(max_iter=1000, random_state=42)
)

# 5. Train model
print("Training model...")
pipeline.fit(X_train, y_train)

# 6. Make predictions
y_train_pred = pipeline.predict(X_train)
y_test_pred = pipeline.predict(X_test)

# 7. Evaluate with comprehensive metrics
print("\n" + "="*50)
print("MODEL EVALUATION METRICS")
print("="*50)

print("\nTRAIN SET METRICS:")
print(f"Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"F1-Score:  {f1_score(y_train, y_train_pred):.4f}")

print("\nTEST SET METRICS:")
print(f"Accuracy:  {accuracy_score(y_test, y_test_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_test_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_test_pred):.4f}")
print(f"F1-Score:  {f1_score(y_test, y_test_pred):.4f}")

print("\nDETAILED CLASSIFICATION REPORT:")
print(classification_report(y_test, y_test_pred, target_names=['Malignant', 'Benign']))

# 8. Save pipeline
print("\n" + "="*50)
print("SAVING PIPELINE")
print("="*50)
joblib.dump(pipeline, "./model/breast_cancer_model.pkl")
print("✓ Pipeline saved as 'breast_cancer_model.pkl'")

# 9. Demonstrate pipeline reloading and prediction
print("\n" + "="*50)
print("DEMONSTRATING PIPELINE RELOADING AND PREDICTION")
print("="*50)

# Reload pipeline
loaded_pipeline = joblib.load("./model/breast_cancer_model.pkl")
print("✓ Pipeline reloaded successfully")

# Make predictions on test set using reloaded pipeline
predictions = loaded_pipeline.predict(X_test)
reloaded_accuracy = accuracy_score(y_test, predictions)
print(f"\nTest accuracy with reloaded pipeline: {reloaded_accuracy:.4f}")

# Example: Make prediction on a new sample
print("\n" + "="*50)
print("EXAMPLE: PREDICTING ON NEW DATA")
print("="*50)
# Use first test sample
sample = X_test.iloc[0:1]
prediction = loaded_pipeline.predict(sample)[0]
probability = loaded_pipeline.predict_proba(sample)[0]

result = "Benign" if prediction == 1 else "Malignant"
print(f"Prediction: {result}")
print(f"Probability - Malignant: {probability[0]:.4f}, Benign: {probability[1]:.4f}")
print("\nSuccess: Pipeline can be reloaded and used for predictions without retraining!")