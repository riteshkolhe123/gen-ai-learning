import pandas as pd 

from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer  
from sklearn.linear_model import BayesianRidge  


df = pd.read_csv('salary_data_impute.csv')
df.info()


imputer = IterativeImputer(estimator= BayesianRidge(), max_iter = 5, random_state = 0)
df['Salary_Iterative'] = imputer.fit_transform(df[['Salary']])


print(df)