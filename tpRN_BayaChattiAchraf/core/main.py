import pandas as pd
import seaborn as sns
import time
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score



#Question 1
data = pd.read_csv("Iris.csv")

#Question 2
print(data.head(10))

#Question 3
print(data.shape)

#Question 4
sns.displot(data= data, x="PetalLengthCm", y="SepalWidthCm")

#Question 5
replace_values = {'Iris-setosa': '0', 'Iris-versicolor': '1', 'Iris-virginica':'2'}
data = data.replace({ 'Species' : replace_values})

#Question 6
print(data.head(10))

#Question 7
X = data[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
Y = data['Species']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3)

#Question 8
#apprentissage
print(Xtrain.head(10))
print(Ytrain.head(10))
#test
print(Xtest.head(10))
print(Ytest.head(10))

#Question 9
clf = MLPClassifier(solver='lbfgs', max_iter=150, tol=0.07, verbose=True)
clf.fit(Xtrain, Ytrain)

#Question 10
start = time.time()
Ypred = clf.predict(Xtest)
print("temps de reponse : %s" % (time.time() - start))
accuracy_score(Ytest, Ypred)

#Question 11
confusion_matrix(Ytest, Ypred)

#Question 15 : x10 fois le nombre fixé
clf2 = MLPClassifier(solver='lbfgs', max_iter=1500, tol=0.07, verbose=True)
clf2.fit(Xtrain, Ytrain)
#calcul du nouveau temps de reponse
start = time.time()
Ypred2 = clf2.predict(Xtest)
print("temps de reponse : %s" % (time.time() - start))
accuracy_score(Ytest, Ypred2)

