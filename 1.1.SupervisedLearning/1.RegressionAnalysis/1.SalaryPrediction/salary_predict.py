import pandas as pd 

import matplotlib.pyplot as plt 

import numpy as np   

data_frame = pd.read_csv('salary_data.csv')

print(data_frame.columns)

print(data_frame.info())

print(data_frame.describe())

correlation = data_frame['Experience'].corr(data_frame['Salary'])
print(correlation)

covariance = np.cov(data_frame['Experience'], data_frame['Salary'])
print(covariance)

print ("Mean salary is : " , data_frame['Salary'].mean())
print ("Median salary is: " ,data_frame['Salary'].median())
print ("Mode salary is: " , data_frame['Salary'].median())

plt.scatter(data_frame['Experience'], data_frame['Salary'])
plt.xlabel('Experience')
plt.ylabel('Salary')
plt.title('Experience vs Salary')
plt.show()
