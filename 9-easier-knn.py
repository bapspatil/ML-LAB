# from MMDN import TALK
from sklearn.datasets import load_iris
iris_dataset=load_iris()

from sklearn.model_selection import train_test_split
# XTrain,XTest,YTrain,YTest = ID["data"] + ID["target"] + random_state=0
X_train, X_test, y_train, y_test = train_test_split(iris_dataset["data"], iris_dataset["target"], random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

from sklearn.metrics import accuracy_score
print(accuracy_score(y_test, y_pred))
