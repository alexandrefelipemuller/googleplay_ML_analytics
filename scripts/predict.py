import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, stratify=y)

print("confiabilidade")
# ======= KNN classifier ========
from sklearn.neighbors import KNeighborsClassifier
bestk,bestscore = 0,0
for k in range(1,100,2):
	clf1 = KNeighborsClassifier(n_neighbors = k, weights='distance', metric='euclidean')
	clf1.fit(X_train,y_train)
	if (clf1.score(X_test, y_test) > bestscore):
		bestk=k
		bestscore=clf1.score(X_test, y_test)
clf1 = KNeighborsClassifier(n_neighbors = bestk)
clf1.fit(X_train,y_train)
print("KNN bestk: "+str(bestk))
print(clf1.score(X_test, y_test))

# ===== SVM ======
#check accuracy of our model on the test data
from sklearn import svm
clf2 = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo',probability=True)
clf2.fit(X_train, y_train)
print("SVM") 
print(clf2.score(X_test, y_test))

#==== RADOM TREE ======
from sklearn.ensemble import RandomForestClassifier
bestn, bestscore = 0,0
for n in range(100,3000,500):
	clf3 = RandomForestClassifier(n_estimators=n)
	clf3 = clf3.fit(X_train, y_train)
	if (clf3.score(X_test, y_test) > bestscore):
                bestn=n
                bestscore=clf3.score(X_test, y_test)
clf3 = RandomForestClassifier(n_estimators=bestn)
clf3 = clf3.fit(X_train, y_train)
print('Random Forest, estimators: '+str(bestn))
print(clf3.score(X_test, y_test))

from sklearn.ensemble import VotingClassifier
eclf = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('rt', clf3)],voting='hard', weights=[1,2,3])
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



def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


#np.set_printoptions(precision=2)

#class_names = [0,1,2,3,4]

# Plot non-normalized confusion matrix
#plot_confusion_matrix(y_test, predicted, classes=class_names, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
#plot_confusion_matrix(y_test, predicted, classes=class_names, normalize=True, title='Normalized confusion matrix')
#plt.show()

f=open('BradescoCartoes.csv')
i=0
f1 = open("sugest.csv", "a")
f2 = open("reclama.csv", "a")
f3 = open("ajuda.csv", "a")
f4 = open("elogio.csv", "a")

lines=f.readlines()
for e in predicted:
	if e==1 or e=='1':
		f1.write(lines[i])
	if e==2 or e=='2':
		f2.write(lines[i])
	if e==3 or e=='3':
		f3.write(lines[i])
	if e==4 or e=='4':	
		f4.write(lines[i])
	i+=1
f1.close()
f2.close()
f3.close()
f4.close()

