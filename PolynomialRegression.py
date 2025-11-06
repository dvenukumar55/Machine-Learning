import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

# Create dataset
X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
y = np.array([20000, 25000, 35000, 55000, 80000])

#  Linear Regression
lin_model = LinearRegression()
lin_model.fit(X, y)

# Polynomial Regression (degree = 4)
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)  # creates new features: [1, x, x^2, x^3, x^4]

poly_model = LinearRegression()
poly_model.fit(X_poly, y)

#  Predictions
X_test = np.linspace(1, 5, 100).reshape(-1, 1)
y_pred_lin = lin_model.predict(X_test)
y_pred_poly = poly_model.predict(poly.transform(X_test))

# Visualization
plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_pred_lin, color='green', label='Linear Fit')
plt.plot(X_test, y_pred_poly, color='red', label='Polynomial Fit (Degree=2)')
plt.xlabel('Experience (Years)')
plt.ylabel('Salary (₹)')
plt.title('Polynomial Regression Example')
plt.legend()
plt.show()
