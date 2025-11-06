#StandardScaler
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pandas as pd

data = {'age': [20, 25, 30, 35, 40],
        'salary': [20000, 30000, 40000, 50000, 60000]}

df = pd.DataFrame(data)
print("Original Data:\n", df)

scaler = StandardScaler()

scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled, columns=df.columns)
print("\nAfter Standard Scaling:\n", scaled_df)


#MinMaxScaler
data = {'age': [20, 25, 30, 35, 40],
        'salary': [20000, 30000, 40000, 50000, 60000]}

df = pd.DataFrame(data)
print("Original Data:\n", df)

scaler = MinMaxScaler()

scaled = scaler.fit_transform(df)

scaled_df = pd.DataFrame(scaled, columns=df.columns)
print("\nAfter Min-Max Scaling:\n", scaled_df)