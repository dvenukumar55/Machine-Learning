import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

model=LinearRegression()
df=pd.read_csv(r"C:\Users\ADMIN\OneDrive\Desktop\PYTHON\Machine Learning\dummies.csv")
print(df)

dummies=pd.get_dummies(df['town'])
print(dummies)

merged=pd.concat([df,dummies],axis='columns')
print(merged)

final=merged.drop(['town','west windsor'],axis='columns')
print(final)

x=final.drop('price',axis='columns')
print(x)
y=final.price
print(y)

model.fit(x,y)
model.predict([[3000,0,0]])

model.score(x,y)

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()

dfle=df
dfle.town=le.fit_transform(dfle.town)
print(dfle)

x=df[['town','area']].values
print(x)

y=dfle.price
print(y)

from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder()

x=ohe.fit_transform(x).toarray()
print(x)

x=x[:,1:]
print(x)
model.fit(x,y)
model.predict([[1,0,2800]])