import pandas as pd   
import matplotlib.pyplot as plt          

df = pd.read_csv('house_prices.csv')

print(df.columns)
print(df.info())

correlation = df['Size'].corr(df['Price'])
print("Correlation is : ",correlation)

plt.plot(df['Size'],df['Price'])
plt.xlabel('Size')
plt.ylabel('Price')
plt.show()