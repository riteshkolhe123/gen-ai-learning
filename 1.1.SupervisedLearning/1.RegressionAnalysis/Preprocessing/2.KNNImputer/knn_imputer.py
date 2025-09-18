import pandas as pd  

from sklearn.impute import KNNImputer 

df = pd.read_csv('salary_data_impute.csv')
df.info()


imputer = KNNImputer(n_neighbors = 3)
df['salary'] = imputer.fit_transform(df[['Salary']])



print(df)