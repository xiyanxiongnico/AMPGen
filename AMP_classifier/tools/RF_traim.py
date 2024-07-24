from sklearn import metrics
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


train_data = pd.read_csv('/media/zzh/data/AMP/train_data/classify/top14train_test.csv', encoding="utf8", index_col=0)
test_data = pd.read_csv('/media/zzh/data/AMP/train_data/classify/top14test_test.csv', encoding="utf8", index_col=0)
x_train0 = train_data.iloc[:, 2:-1].values
x_test = test_data.iloc[:, 2:-1].values
y_train0 = train_data.iloc[:, -1].values
y_test = test_data.iloc[:, -1].values

randomforest = RandomForestClassifier(random_state = 42, n_estimators = 2000)
#Train the classifier model using the training set samples
randomforest.fit(x_train0,y_train0)
y_pred = randomforest.predict(x_test)

y_true = y_test
F1 = metrics.f1_score(y_true, y_pred)
accuracy = metrics.accuracy_score(y_true, y_pred)
recall  = metrics.recall_score(y_true, y_pred)
print(f"F1:{F1}|Accuracy:{accuracy}|Recall:{recall}")


