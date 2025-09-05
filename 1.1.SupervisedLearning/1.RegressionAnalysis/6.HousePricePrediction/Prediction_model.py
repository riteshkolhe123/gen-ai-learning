import pandas as pd    
from sklearn.preprocessing import PolynomialFeatures  
from sklearn.linear_model import LinearRegression 

df = pd.read_csv('house_prices.csv')

x = df.drop('Price',axis = 1)
y = df['Price']

poly = PolynomialFeatures(degree = 3)
x_square = poly.fit_transform(x)  

model = LinearRegression()

model.fit(x_square,y)

price = model.predict(poly.fit_transform([[1200]]))
print("Price of house is : ",price)