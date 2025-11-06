import pandas as pd
from sklearn.model_selection import train_test_split

data = {
    'area': [1000, 1500, 2000, 2500, 3000],
    'bedrooms': [2, 3, 3, 4, 4],
    'age': [10, 5, 8, 3, 1],
    'price': [50, 60, 70, 85, 95]  
}

df = pd.DataFrame(data)

X = df[['area', 'bedrooms', 'age']]   
y = df['price']                      

print("Features (X):")
print(X)

print("\nTarget (y):")
print(y)

# Step 4: Split data into training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("\nTraining Features:")
print(X_train)

print("\nTesting Features:")
print(X_test)

print("\nTraining Targets:")
print(y_train)

print("\nTesting Targets:")
print(y_test)