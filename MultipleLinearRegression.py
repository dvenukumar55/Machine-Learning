import pandas as pd
from sklearn.linear_model import LinearRegression

#  Create a dataset
data = {
    'experience': [1, 2, 3, 4, 5],
    'age': [22, 25, 27, 30, 35],
    'salary': [20000, 25000, 30000, 35000, 40000]
}
df = pd.DataFrame(data)

#  Split Features (X) and Target (y)
X = df[['experience', 'age']]   # Independent variables
y = df['salary']                # Dependent variable

# Create and Train the model
model = LinearRegression()
model.fit(X, y)

# Display learned coefficients
print("Intercept (b0):", model.intercept_)
print("Coefficients (b1, b2):", model.coef_)

#  Predict salary for new data
predicted_salary = model.predict([[6, 40]])  # 6 years exp, age 40
print("\nPredicted Salary for 6 yrs exp & age 40:", predicted_salary[0])

#  Visualize the results
y_pred = model.predict(X)
df['predicted_salary'] = y_pred

print(df)