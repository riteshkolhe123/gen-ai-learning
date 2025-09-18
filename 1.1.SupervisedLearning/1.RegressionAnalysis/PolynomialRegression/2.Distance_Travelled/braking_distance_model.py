import pandas as pd         

from sklearn.linear_model import LinearRegression        
from sklearn.preprocessing import PolynomialFeatures 
import pickle

df = pd.read_csv('distance.csv')

x = df.drop('BrakingDistance',axis = 1)
y = df['BrakingDistance']

poly = PolynomialFeatures(degree = 5)
speed_square = poly.fit_transform(x) 

model = LinearRegression()

model.fit(speed_square,y)

with open('Speed_Prediction.','wb') as file:
    pickle.dump(model.file)

braking_distance = model.predict(poly.fit_transform([[25]]))

print("Braking Distance is : ", braking_distance)