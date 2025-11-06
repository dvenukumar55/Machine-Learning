#Detecting Outliers
import pandas as pd

data = {'salary': [50000, 52000, 54000, 56000, 58000, 60000, 200000,2345678]}  
df = pd.DataFrame(data)
print("Original Data:")
print(df)

Q1 = df['salary'].quantile(0.25)
Q3 = df['salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['salary'] < lower) | (df['salary'] > upper)]
print("\nOutliers Detected:")
print(outliers)


#Removing Outliers
clean_df = df[(df['salary'] >= lower) & (df['salary'] <= upper)]

print("\nAfter Removing Outliers:")
print(clean_df)