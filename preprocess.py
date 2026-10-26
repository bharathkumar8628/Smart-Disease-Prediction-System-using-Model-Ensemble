from ucimlrepo import fetch_ucirepo
import pandas as pd

# --------------------------------
# 1. Download the dataset
# --------------------------------

heart_disease = fetch_ucirepo(id=45)

# Get features and target
X = heart_disease.data.features
y = heart_disease.data.targets

# Combine them into one DataFrame
data = pd.concat([X, y], axis=1)

print("Original Dataset Shape:")
print(data.shape)

print("\nOriginal Data:")
print(data.head())


# --------------------------------
# 2. Check missing values
# --------------------------------

print("\nMissing Values:")
print(data.isnull().sum())


# --------------------------------
# 3. Convert columns to numeric
# --------------------------------

for column in data.columns:
    data[column] = pd.to_numeric(data[column], errors="coerce")


# --------------------------------
# 4. Handle missing values
# --------------------------------

data = data.dropna()

print("\nDataset Shape After Removing Missing Values:")
print(data.shape)


# --------------------------------
# 5. Convert target into binary
# --------------------------------

# 0 = No Disease
# 1,2,3,4 = Disease

data["num"] = data["num"].apply(
    lambda value: 0 if value == 0 else 1
)


# --------------------------------
# 6. Display final dataset
# --------------------------------

print("\nFinal Dataset:")
print(data.head())

print("\nTarget Distribution:")
print(data["num"].value_counts())


# --------------------------------
# 7. Save cleaned dataset
# --------------------------------

data.to_csv("data/heart_disease_cleaned.csv", index=False)

print("\nCleaned dataset saved successfully!")