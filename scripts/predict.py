
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

print("confiabilidade")
# ======= KNN classifier ========
from sklearn.neighbors import KNeighborsClassifier
k=7
clf1 = KNeighborsClassifier(n_neighbors = k)
clf1.fit(X_train,y_train)
#print(clf1.score(X_test, y_test))

# ===== SVM ======
#check accuracy of our model on the test data
from sklearn import svm
clf2 = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo',probability=True)
clf2.fit(X_train, y_train) 
#print(clf2.score(X_test, y_test))

#==== RADOM TREE ======
from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=100)
clf3 = clf3.fit(X_train, y_train)
#print(clf3.score(X_test, y_test))

from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('rt', clf3)],voting='hard', weights=[1, 2, 2])
eclf = eclf.fit(X_train, y_train)
print(eclf.score(X_test, y_test))

def showClassifierDist(predicted):
	print('distribuicao no projeto')
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

df2 = pd.read_csv("output_p.csv")
X_predict = df2.drop(columns=['target'])
predicted = eclf.predict(X_predict)
showClassifierDist(predicted)

#f=open('BradescoCartoes.csv')
#i=0
#f1 = open("sugest.csv", "a")
#f2 = open("reclama.csv", "a")
#f3 = open("ajuda.csv", "a")
#f4 = open("elogio.csv", "a")
#
#lines=f.readlines()
#for e in predicted:
#	if e==1 or e=='1':
#		f1.write(lines[i])
#	if e==2 or e=='2':
#		f2.write(lines[i])
#	if e==3 or e=='3':
#		f3.write(lines[i])
#	if e==4 or e=='4':	
#		f4.write(lines[i])
#	i+=1
#f1.close()
#f2.close()
#f3.close()
#f4.close()

