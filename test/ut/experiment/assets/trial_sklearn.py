import nni

from sklearn import datasets
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

iris = datasets.load_iris()

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.33)

params = nni.get_next_parameter()
clf = SVC(**params)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
nni.report_final_result(accuracy_score(y_test, y_pred))
