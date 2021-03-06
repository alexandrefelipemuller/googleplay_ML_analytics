import numpy as np
import random
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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
#X_train = X
#y_train = y
#df2 = pd.read_csv("output_p.csv")
#y_test = df2['target']
#X_test = df2.drop(columns=['target'])

nb_classes = len(np.unique(y_test))
 
print("confiabilidade")
# ======= KNN classifier ========
from sklearn.neighbors import KNeighborsClassifier
clf1 = KNeighborsClassifier(n_neighbors = 17, weights='distance', metric='euclidean')
clf1.fit(X_train,y_train)
#print(clf1.score(X_test, y_test))

# ===== SVM ======
#check accuracy of our model on the test data
from sklearn import svm
clf2 = svm.SVC(gamma='scale', kernel='linear', decision_function_shape='ovo',probability=True)
clf2.fit(X_train, y_train)
#print("SVM") 
#print(clf2.score(X_test, y_test))

#==== RADOM TREE ======
from sklearn.ensemble import RandomForestClassifier
clf3 = RandomForestClassifier(n_estimators=600)
clf3 = clf3.fit(X_train, y_train)
#print(clf3.score(X_test, y_test))

from sklearn.ensemble import VotingClassifier

eclf = VotingClassifier(estimators=[('knn', clf1), ('svm', clf2), ('rt', clf3)],voting='hard', weights=[2,2,5])
eclf = eclf.fit(X_train, y_train)
print(str(round((eclf.score(X_test, y_test))*100,2)) + "%")

print("Checking models...")

def showClassifierDist(predicted, nb_classes):
	print('distribuicao no projeto')
	classes = [0] * nb_classes
	for e in predicted:
	    for cl in range(nb_classes):
		if int(e)==(int(cl)+1):
        		classes[cl]+=1
	print(classes)

df2 = pd.read_csv("output_p.csv")
X_predict = df2.drop(columns=['target'])
predicted = eclf.predict(X_predict)
showClassifierDist(predicted, nb_classes)

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


np.set_printoptions(precision=2)
class_names = range(1,nb_classes+1)

# Plot non-normalized confusion matrix
#plot_confusion_matrix(y_test, predicted, classes=class_names, title='Confusion matrix, without normalization')
# Plot normalized confusion matrix
x_predicted = eclf.predict(X_test)
plot_confusion_matrix(y_test, x_predicted, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()

splitfile = sys.argv[1]
if len(splitfile) > 1:
	f=open(splitfile)
	i=0
	filesd = []
	for cl in range(nb_classes):
		filesd.append(open(str(cl+1)+".csv", "w"))
	
	lines=f.readlines()
	for e in predicted:
		for cl in range(nb_classes):
			if int(e) == cl+1:
				filesd[cl].write(lines[i])
		i+=1
	
	for cl in range(nb_classes):
		filesd[cl].close()
	
