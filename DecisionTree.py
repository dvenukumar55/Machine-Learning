
#                 ┌───────────────┐
#                 │ Age < 30 ?     │
#                 └──────┬────────┘
#                        │
#         ┌──────────────┴──────────────┐
#    Yes (young)                    No (old)
#  Income < 50k ?                   → Buy = No
#       │
#   ┌───┴───────┐
# Yes → No     No → Yes


import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt

#  Create dataset
data = {
    'hours_studied': [1, 2, 3, 4, 5, 6, 7, 8],
    'sleep_hours': [8, 7, 6, 5, 5, 4, 3, 2],
    'passed': [0, 0, 0, 0, 1, 1, 1, 1]
}
df = pd.DataFrame(data)

#  Split features and target
X = df[['hours_studied', 'sleep_hours']]
y = df['passed']

#  Create and Train Model
model = DecisionTreeClassifier(criterion='entropy', random_state=42)
model.fit(X, y)

#  Predict for new data
new_data = [[4, 6]]
prediction = model.predict(new_data)

print("Predicted (0=Fail, 1=Pass):", prediction[0])

# Step 6: Visualize the Tree
plt.figure(figsize=(8, 6))
plot_tree(model, feature_names=['hours_studied', 'sleep_hours'], 
          class_names=['Fail', 'Pass'], filled=True)
plt.show()
