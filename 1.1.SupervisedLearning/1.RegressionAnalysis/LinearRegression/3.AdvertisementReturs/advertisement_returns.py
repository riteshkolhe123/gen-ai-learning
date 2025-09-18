import pandas as pd  

import matplotlib.pyplot as plt    

import numpy as np   

df = pd.read_csv('advertising.csv')

print(df.columns)

print(df.info())

print(df.describe())

correlation_TV_sales = df['TV'].corr(df['sales'])
print("Correlation in TV and Sales is : ",correlation_TV_sales)

covariance = np.cov(df['TV'], df['sales'])
print("Covariance of TV and Sales is : ",covariance)

plt.scatter(df['sales'],df['TV'])
plt.xlabel('sales')
plt.ylabel('TV')
plt.title('TV vs Sales')
plt.show()

correlation_radio_sales = df['radio'].corr(df['sales'])
print("Correlation of radio and sales is : ",correlation_radio_sales )

covariance_radio_sales = np.cov(df['radio'],df['sales'])
print("covariance of radio and sales is : ",covariance_radio_sales)

plt.scatter(df['radio'],df['sales'])
plt.xlabel('sales')
plt.ylabel('radio')
plt.title('Radio vs Sales ')
plt.show()

correlation_newspaper_sales = df['newspaper'].corr(df['sales'])
print("Correlation of newspaper and sales is: ",correlation_newspaper_sales)

covariance_newspaper_sales = np.cov(df['newspaper'],df['sales'])
print("Covariance of newspaper and sales is :",covariance_newspaper_sales)

plt.scatter(df['newspaper'],df['sales'])
plt.xlabel('sales')
plt.ylabel('newspaper')
plt.title('Newspaper vs Sales')
plt.show()