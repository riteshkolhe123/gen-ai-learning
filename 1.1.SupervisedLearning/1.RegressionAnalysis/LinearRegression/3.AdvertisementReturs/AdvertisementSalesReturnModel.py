import pandas as pd    

from sklearn.linear_model import LinearRegression   

df = pd.read_csv('advertising.csv')

x = df.drop('sales',axis = 1)

y = df['sales']

model = LinearRegression()

model.fit(x,y)

m = model.coef_[0]
print("Coefficient is : ",m)   

c = model.intercept_  
print("Intercept is : ",c)   

sales = model.predict(pd.DataFrame([[100,40,60],[100.5,40.5,60.5]], columns = ['TV','radio','newspaper']))
print("When the amount of TV is 100, Radio is 40, Newspaper is 60, the sale in lakh is : ",sales[0])
print("When the amount of TV is 100.5, Radio is 40.5, Newspaper is 60.5, the sale in lakh is : ",sales[1])