import pandas as pd
data = {
    'name': ['Ravi', 'Rani', 'Arjun', 'Kittu'],
    'age': [25, None, 30, 28],
    'salary': [50000, 60000, None, 58000]
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

# 1. Fill missing numeric values with mean
df['age'].fillna(df['age'].mean(), inplace=True)
df['salary'].fillna(df['salary'].mean(), inplace=True)

print("\nAfter Filling Missing Values by Mean:")
print(df)

# 2. Fill missing numeric values with median
df['age'].fillna(df['age'].median(), inplace=True)
df['salary'].fillna(df['salary'].median(), inplace=True)

print("\nAfter Filling Missing Values by Median:")
print(df)

# 3. Fill missing numeric values with mode
df['age'].fillna(df['age'].mode()[0], inplace=True)
df['salary'].fillna(df['salary'].mode()[0], inplace=True)

print("\nAfter Filling Missing Values by Mode:")
print(df)

# 4. Drop rows with missing values
df_dropped = df.dropna()
print("\nAfter Dropping Rows with Missing Values:")
print(df_dropped)

#5.fill missing values using forward fill
df_ffill = df.fillna(method='ffill')
print("\nAfter Forward Fill:")
print(df_ffill)

#6.fill missing values using backward fill
df_bfill = df.fillna(method='bfill')
print("\nAfter Backward Fill:")
print(df_bfill)