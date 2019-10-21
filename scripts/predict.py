
import pandas as pd
import sys

#read in the data using pandas

df = pd.read_csv("output_m.csv")
#check data has been read in properly
#print(df.head())

#check number of rows and columns in dataset
#print(df.shape)
#create a dataframe with all training data except the target column

X = df.drop(columns=['target'])
#check that the target variable has been removed
#print(X.head())

#separate target values
y = df['target'].values

#view target values
#print(y[0:5])

from sklearn.model_selection import train_test_split
#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=1, stratify=y)

from sklearn.neighbors import KNeighborsClassifier
# Create KNN classifier
k=7
knn = KNeighborsClassifier(n_neighbors = k)
# Fit the classifier to the data
knn.fit(X_train,y_train)

#check accuracy of our model on the test data
print("confiabilidade")
print(knn.score(X_test, y_test))

print('distribuicao no projeto')
df2 = pd.read_csv("output_p.csv")
X_predict = df2.drop(columns=['target'])
predicted = knn.predict(X_predict)

classe1,classe2,classe3,classe4 = 0,0,0,0
for e in predicted:
    if e==1 or e=='1':
        classe1+=1
    if e==2 or e=='2':
        classe2+=1
    if e==3 or e=='3':
        classe3+=1
    if e==4 or e=='4':
        classe4+=1
print(classe1, classe2, classe3, classe4)


