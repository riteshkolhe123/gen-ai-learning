import pandas as pd    
import matplotlib.pyplot as plt     

df = pd.read_csv('distance.csv')

print(df.info())

correlation = df['Speed'].corr(df['BrakingDistance'])
print("correlationis: ",correlation)

plt.plot(df['Speed'],df['BrakingDistance'])
plt.xlabel('Speed')
plt.ylabel('BrakingSpeed')
plt.title('Speed vs BrakingDistance')
plt.show()