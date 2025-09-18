import pandas as pd   
from sklearn.linear_model import LinearRegression

from sklearn.preprocessing import LabelEncoder  

df = pd.read_csv('salary_data_labelEnco.csv')
df.info()
x = df.drop('Salary',axis = 1)
y = df['Salary']


encoder = LabelEncoder()
x['Title'] = encoder.fit_transform(x['Title'])

print(x)


model = LinearRegression()
model.fit(x,y)
salary = model.predict(pd.DataFrame([[1,3]],columns =['Title','Experience']))
print(salary)

