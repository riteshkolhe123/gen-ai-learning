import pandas as pd    
import numpy as nm   
from sklearn.linear_model import LinearRegression   
from sklearn.model_selection import train_test_split   

df = pd.read_csv("SpendingData.csv")

x = df.drop('Spendings', axis = 1)
y = df['Spendings']

x_train, x_test, y_train, y_test = train_test_split(x,y , test_size = 0.2, random_state = 100)

# print("X train head is : ",x_train.head(),"\n")
# print("X test head is : ",x_test.head(),"\n")
# print("Y train head is : ",y_train.head(),"\n")
# print("Y test head is : ",y_test.head())

model = LinearRegression()
model.fit(x_train,y_train)

train_score = model.score(x_train,y_train)
test_score = model.score(x_test,y_test)

print("Training Score is : ",train_score)
print("Testing Score is :",test_score)

output = model.predict(pd.DataFrame([[20,37541]], columns = ['Age','Salary']))

print(output)
