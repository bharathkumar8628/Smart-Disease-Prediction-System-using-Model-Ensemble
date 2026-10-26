import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# --------------------------------
# 1. Load cleaned dataset
# --------------------------------

data = pd.read_csv("data/heart_disease_cleaned.csv")

print("Dataset Shape:")
print(data.shape)


# --------------------------------
# 2. Separate features and target
# --------------------------------

X = data.drop("num", axis=1)

y = data["num"]


print("\nFeatures (X):")
print(X.head())

print("\nTarget (y):")
print(y.head())


# --------------------------------
# 3. Display feature names
# --------------------------------

print("\nFeature Names:")
print(X.columns.tolist())


# --------------------------------
# 4. Split dataset
# --------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)


# --------------------------------
# 5. Feature Scaling
# --------------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)

X_test_scaled = scaler.transform(X_test)


print("\nScaled Training Data:")
print(X_train_scaled[:5])

print("\nScaled Testing Data:")
print(X_test_scaled[:5])


# --------------------------------
# 6. Final information
# --------------------------------

print("\nModel preparation completed successfully!")