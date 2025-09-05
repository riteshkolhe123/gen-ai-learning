import pandas as pd   
from sklearn.linear_model import LinearRegression   
from sklearn.preprocessing import PolynomialFeatures

df = pd.read_csv('data.csv')

x = df.drop('b',axis = 1)
y = df['b']

poly = PolynomialFeatures(degree = 2)

x_square = poly.fit_transform(x)

model = LinearRegression()

model.fit(x_square,y)

square = model.predict(poly.fit_transform([[6]]))

print("Square is : ",square)
