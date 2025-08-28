import pandas as pd   

from sklearn.linear_model import LinearRegression   

df = pd.read_csv('car_price_data.csv')

x = df.drop('Price', axis = 1)

y = df['Price']

model = LinearRegression()

model.fit(x,y)

m = model.coef_[0]
print('Coeficient is : ',m)  

c = model.intercept_  
print('Intercept is : ',c)  

CarPrice = model.predict(pd.DataFrame([[9,25000],[10,25000],[11,25000]], columns = ['Age','Mileage']))

print("The price of car having age 9 years and mileage 25000 km is : ",CarPrice[0])
print("The price of car having age 10 years and mileage 25000 km is : ",CarPrice[1])
print("The price of car having age 11 years and mileage 25000 km is : ",CarPrice[2])