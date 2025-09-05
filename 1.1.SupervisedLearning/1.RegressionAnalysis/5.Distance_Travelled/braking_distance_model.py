import pandas as pd         

from sklearn.linear_model import LinearRegression        
from sklearn.preprocessing import PolynomialFeatures 

df = pd.read_csv('distance.csv')

x = df.drop('BrakingDistance',axis = 1)
y = df['BrakingDistance']

poly = PolynomialFeatures(degree = 3)
speed_square = poly.fit_transform(x) 

model = LinearRegression()

model.fit(speed_square,y)

coefficient = model.coef_[0]
print("coefficient is : ", coefficient)

intercept = model.intercept_
print("Intercept is : ", intercept)

braking_distance = model.predict(poly.fit_transform([[25]]))

print("Braking Distance is : ", braking_distance)