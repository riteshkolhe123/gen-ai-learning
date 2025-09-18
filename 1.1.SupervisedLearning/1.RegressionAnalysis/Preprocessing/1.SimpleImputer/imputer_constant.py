import pandas as pd   
from sklearn.impute import SimpleImputer 

df = pd.read_csv('salary_data_impute.csv')

df.info()

imputer = SimpleImputer(strategy='constant',fill_value = 25000.269)
df['Salary'] = imputer.fit_transform(df[['Salary']])

print(df)
df.info()