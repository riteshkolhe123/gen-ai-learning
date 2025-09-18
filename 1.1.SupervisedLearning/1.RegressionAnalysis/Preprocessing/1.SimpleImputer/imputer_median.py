import pandas as pd   
from sklearn.impute import SimpleImputer

df = pd.read_csv('salary_data_impute.csv')

df.info()

imputer = SimpleImputer(strategy='median')

df['Salary_imputed'] = imputer.fit_transform(df[['Salary']])

print(df)