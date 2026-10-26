from ucimlrepo import fetch_ucirepo
import pandas as pd

# Download Heart Disease dataset
heart_disease = fetch_ucirepo(id=45)

# Get features
X = heart_disease.data.features

# Get target
y = heart_disease.data.targets

# Combine features and target
data = pd.concat([X, y], axis=1)

# Display basic information
print("Dataset Shape:")
print(data.shape)

print("\nFirst 5 Rows:")
print(data.head())

print("\nColumn Names:")
print(data.columns)

print("\nMissing Values:")
print(data.isnull().sum())