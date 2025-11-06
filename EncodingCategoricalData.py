# Encoding Categorical Data

from sklearn.preprocessing import LabelEncoder
import pandas as pd

data = {
    'country': ['India', 'USA', 'UK', 'India', 'UK'],
    'purchased': ['Yes', 'No', 'Yes', 'No', 'Yes']
}

df = pd.DataFrame(data)
print("Original Data:")
print(df)

le_country = LabelEncoder()
le_purchase = LabelEncoder()

df['country'] = le_country.fit_transform(df['country'])
df['purchased'] = le_purchase.fit_transform(df['purchased'])

print("\nAfter Label Encoding:")
print(df)

# One-Hot Encoding

import pandas as pd

data = {'country': ['India', 'USA', 'UK','India']}
df = pd.DataFrame(data)

encoded_df = pd.get_dummies(df, columns=['country'], drop_first=True)
print(encoded_df)