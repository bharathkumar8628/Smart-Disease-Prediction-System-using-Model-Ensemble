import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------
# 1. Load cleaned dataset
# --------------------------------

data = pd.read_csv("data/heart_disease_cleaned.csv")

print("Dataset Shape:")
print(data.shape)

print("\nDataset Information:")
print(data.info())

print("\nStatistical Summary:")
print(data.describe())


# --------------------------------
# 2. Disease Distribution
# --------------------------------

plt.figure(figsize=(6, 4))

sns.countplot(x="num", data=data)

plt.title("Heart Disease Distribution")
plt.xlabel("Disease Status")
plt.ylabel("Number of Patients")

plt.xticks(
    [0, 1],
    ["No Disease", "Disease"]
)

plt.tight_layout()
plt.show()


# --------------------------------
# 3. Age Distribution
# --------------------------------

plt.figure(figsize=(7, 4))

sns.histplot(
    data=data,
    x="age",
    bins=15,
    kde=True
)

plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Number of Patients")

plt.tight_layout()
plt.show()


# --------------------------------
# 4. Correlation Heatmap
# --------------------------------

plt.figure(figsize=(12, 8))

correlation = data.corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Feature Correlation Heatmap")

plt.tight_layout()
plt.show()