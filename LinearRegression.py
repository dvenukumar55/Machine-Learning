import pandas as pd
from sklearn.linear_model import LinearRegression

#  Create a simple dataset
data = {
    'experience': [1, 2, 3, 4, 5],
    'salary': [20000, 25000, 30000, 35000, 40000]
}
df = pd.DataFrame(data)

#  Separate Features (X) and Target (y)
X = df[['experience']]  # Feature
y = df['salary']        # Target

#  Create and train the model
model = LinearRegression()
model.fit(X, y)

#  Predict for new data
predicted_salary = model.predict([[10]])

print("Predicted salary for 10 years experience:", predicted_salary[0])

print("Slope (m):", model.coef_)
print("Intercept (c):", model.intercept_)
#y=mx+c:Salary=5000×Experience+15000


# Optional: Visualize the results
import matplotlib.pyplot as plt

# Plot data points
plt.scatter(X, y, color='blue', label='Actual Data')

# Plot regression line
plt.plot(X, model.predict(X), color='red', label='Best Fit Line')

plt.xlabel('Experience (Years)')
plt.ylabel('Salary (₹)')
plt.title('Simple Linear Regression')
plt.legend()
plt.show()