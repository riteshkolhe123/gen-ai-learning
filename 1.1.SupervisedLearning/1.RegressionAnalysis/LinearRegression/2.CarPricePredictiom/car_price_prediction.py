import pandas as pd   

import matplotlib.pyplot as plt    

import numpy as np   

df = pd.read_csv('car_price_data.csv')

print(df.columns)

print(df.info())

print(df.describe())

correlation_Age_Price = df['Age'].corr(df['Price'])
print("Correlation of Age and Price is :",correlation_Age_Price)

covariance_Age_Price = np.cov(df['Age'],df['Price'])
print("covariance of Age and Price is :",covariance_Age_Price)

plt.scatter(df['Age'],df['Price'])
plt.xlabel('price')
plt.ylabel('Age')
plt.title('Age vs Price')   
plt.show()

correlation_Mileage_Price = df['Mileage'].corr(df['Price'])
print("Correlation of Mileage and price is :",correlation_Mileage_Price)

covariance_Mileage_Price = np.cov(df['Mileage'],df['Price'])
print("Covariance of Mileage and Price is : ",covariance_Mileage_Price)

plt.scatter(df['Mileage'],df['Price'])
plt.xlabel('price')
plt.ylabel('Mileage')
plt.title('Mileage vs Price')
plt.show()
