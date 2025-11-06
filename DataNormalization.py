
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = {
    'age': [20, 25, 30, 35, 40],
    'salary': [20000, 30000, 40000, 50000, 60000]
}

df = pd.DataFrame(data)
print("Original Data:\n", df)

# Apply Normalization
scaler = MinMaxScaler()
normalized = scaler.fit_transform(df)

normalized_df = pd.DataFrame(normalized, columns=df.columns)
print("\nAfter Normalization:\n", normalized_df)


#Outlier handling + Normalization

data = {'salary': [30000, 35000, 40000, 45000, 50000, 100000]}
df = pd.DataFrame(data)

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

clean_df = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

scaler = MinMaxScaler()
normalized = scaler.fit_transform(clean_df[['salary']])
normalized_df = pd.DataFrame(normalized, columns=['salary_normalized'])

print("Original Data:\n", df)
print("\nCleaned Data (Outliers Removed):\n", clean_df)
print("\nNormalized Data:\n", normalized_df)