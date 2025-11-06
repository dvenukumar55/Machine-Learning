# We want to predict whether a student 
# passes (1) or fails (0) based on hours studied

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

# Create a simple dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'passed': [0, 0, 0, 0, 1, 1, 1, 1]  
}
df = pd.DataFrame(data)

# Split features (X) and target (y)
X = df[['hours_studied']]
y = df['passed']

# Create and train the model
model = LogisticRegression()
model.fit(X, y)

# Predict for a new value
new_hours = [[12]]
prediction = model.predict(new_hours)
probability = model.predict_proba(new_hours)

print("Predicted Class (0=Fail, 1=Pass):", prediction[0])
print("Probability [Fail, Pass]:", probability[0])

# Visualize the results
X_test = np.linspace(0, 8, 100).reshape(-1, 1)
y_prob = model.predict_proba(X_test)[:, 1]  # probability of class 1

plt.scatter(X, y, color='blue', label='Actual Data')
plt.plot(X_test, y_prob, color='red', label='Sigmoid Curve')
plt.xlabel('Hours Studied')
plt.ylabel('Probability of Passing')
plt.title('Logistic Regression')
plt.legend()
plt.show()

# Binary Logistic Regression	2 categories	Pass/Fail, Yes/No
# Multinomial Logistic Regression	3+ categories	Predicting fruit type (apple/orange/banana)
# Ordinal Logistic Regression	Ordered categories	Low/Medium/High 