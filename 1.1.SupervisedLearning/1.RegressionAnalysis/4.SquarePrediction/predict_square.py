import pandas as pd   
import matplotlib.pyplot as plt      
import numpy as np    

df = pd.read_csv('data.csv')

print(df.columns)
df.info()

correlation = df['a'].corr(df['b'])
print("correlation is : ", correlation)

plt.plot(df['a'],df['b'])
plt.xlabel('a')
plt.ylabel('b')
plt.show()