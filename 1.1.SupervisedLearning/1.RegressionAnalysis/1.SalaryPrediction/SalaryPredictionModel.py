import pandas as pd   

from sklearn.linear_model import LinearRegression 

import numpy as np   

df = pd.read_csv('salary_data.csv')

x = df.drop('Salary', axis = 1)     

y = df['Salary']                      

model = LinearRegression()            
model.fit(x,y)

m = model.coef_[0]
print("Coeficient is :",m)

c = model.intercept_
print("Intercept is : ",c)

salaries = model.predict(pd.DataFrame([[20],[21],[22]], columns = ['Experience']))

print("Salary of employee when experiance is 20 years is : ", salaries[0])
print("Salary of employee when experiance is 21 years is : ", salaries[1])
print("Salary of employee when experiance is 22 years is : ", salaries[2])