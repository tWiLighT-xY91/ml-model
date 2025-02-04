import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


# Load dataset
df = pd.read_csv(r"c:\Users\lenovo\Cancer_Data.csv")

# Verify data loading
print("Dataset Shape:", df.shape)
print(df.head())  # Check first few rows

# Handle missing values by filling them with median
df.fillna(df.median(numeric_only=True), inplace=True)

# Drop unnecessary columns safely
df = df.drop(columns=[col for col in ["id", "Unnamed: 32"] if col in df.columns], errors="ignore")

# Ensure 'diagnosis' column is clean
if "diagnosis" in df.columns:
    df["diagnosis"] = df["diagnosis"].astype(str).str.strip()
    df["diagnosis"] = df["diagnosis"].map({"M": 1, "B": 0})  # Convert 'M' to 1, 'B' to 0
    df = df.dropna(subset=["diagnosis"])  # Drop NaN values

# Define features (X) and target (y)
if "diagnosis" in df.columns:
    X = df.drop(columns=["diagnosis"])
    y = df["diagnosis"]
else:
    raise ValueError("Column 'diagnosis' not found in dataset!")

# Ensure X and y are not empty
print("X shape:", X.shape)  # Should NOT be (0, ...)
print("y shape:", y.shape)  # Should NOT be (0, )
if X.shape[0] == 0 or y.shape[0] == 0:
    raise ValueError("Error: Features or labels are empty. Check data preprocessing steps.")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training Set Shape:", X_train.shape)
print("Testing Set Shape:", X_test.shape)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression Model
model = LogisticRegression()
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Check accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Classification Report
print("Classification Report:\n", classification_report(y_test, y_pred))


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)


model = LogisticRegression()
model.fit(X_train_scaled, y_train)


joblib.dump(model, "lung_cancer_model.pkl")


joblib.dump(scaler, "scaler.pkl")

print("Model and Scaler saved successfully!")