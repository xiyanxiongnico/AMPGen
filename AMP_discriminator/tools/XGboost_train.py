import xgboost
from sklearn import metrics
import pandas as pd
from sklearn.feature_selection import SelectFromModel
import numpy as np
from sklearn.model_selection import  cross_val_score, KFold,train_test_split
from tqdm import tqdm



data = pd.read_csv('/Users/zhangzheng/Desktop/xgboost_classifier/data_classify/top14Featured_all.csv')
X = data.iloc[:, 2:-1].values
Y = data.iloc[:, -1].values

#trainning model
learning_rate = [0.01,0.05,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
max_depth = [4, 5, 6, 7, 8, 9, 10,11,12,13,14,15]
n_estimatorsd = [50,100,200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000]
best_score = 0
lr_p = []
md_p = []
ne_p = []
auc_p = []
ac_p = []
f1_p = []
for lr in learning_rate:
    for md in max_depth:
        for ne in n_estimatorsd:
            model = xgboost.XGBClassifier(max_depth=md, n_estimators=ne, learning_rate=lr, objective="binary:logistic",tree_method="hist", device="cuda")
            kflod = KFold(n_splits=5, random_state=7, shuffle=True)
           

            auc =  cross_val_score(model, X, Y, cv=kflod, scoring='roc_auc')
            # print(auc)
            accuracy  = cross_val_score(model, X, Y, cv=kflod, scoring='accuracy')
            # print(accuracy)
            recall  = cross_val_score(model, X, Y, cv=kflod, scoring='recall')
            # print(recall)
            F1 = cross_val_score(model, X, Y, cv=kflod, scoring='f1')
            # print(F1)
            precision= cross_val_score(model, X, Y, cv=kflod, scoring='precision')
            print(precision)
            lr_p.append(lr)
            md_p.append(md)
            ne_p.append(ne)
            auc_p.append(auc.mean())
            ac_p.append(accuracy.mean())
            f1_p.append(F1.mean())
            score = auc.mean()

            if  score > best_score:
                best_score = score
                print("best_auc:", best_score)
                best_parameters = {'learning_rate': lr, "max_depth": md, "n_estimators": ne}
                print("best_parameters:", best_parameters)
# defined best model 
model = xgboost.XGBClassifier(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"],
                                learning_rate=best_parameters["learning_rate"], 
                                objective="binary:logistic",tree_method="hist", device="cuda")
kflod = KFold(n_splits=5, random_state=7, shuffle=True)


auc =  cross_val_score(model, X, Y, cv=kflod, scoring='roc_auc')
accuracy  = cross_val_score(model, X, Y, cv=kflod, scoring='accuracy')
recall  = cross_val_score(model, X, Y, cv=kflod, scoring='recall')
F1 = cross_val_score(model, X, Y, cv=kflod, scoring='f1')
precision= cross_val_score(model, X, Y, cv=kflod, scoring='precision')
print(f"F1:{F1}|Accuracy:{accuracy}|Recall:{recall}|AUC:{auc}")

#save all the paraments
df = pd.DataFrame({'learning_rate':lr_p,'max_depth':md_p,'n_estimatorsd':ne_p,'AUC':auc_p,'F1':f1_p,'Accuracy':ac_p})
df.to_csv('/media/zzh/data/AMP/code_available/result/paramenters.csv')


# choosing feature

x_train0,x_test,y_train0,y_test = train_test_split(X,Y,test_size = 0.3, random_state= 77)
model = xgboost.XGBClassifier(max_depth=best_parameters["max_depth"], n_estimators=best_parameters["n_estimators"],
                                learning_rate=best_parameters["learning_rate"], 
                                objective="binary:logistic",tree_method="hist", device="cuda")
model.fit(x_train0, y_train0)
thresholds = np.sort(model.feature_importances_)
print(thresholds)
thresh_list = []
n_list = []
accuracy_list = []
F1_list = []
recall_list = []
for thresh in tqdm(thresholds):
    # select features using threshold
        selection = SelectFromModel(model,threshold=thresh,prefit=True )
        select_X_train = selection.transform(x_train0)
        # train model
        selection_model = xgboost.XGBClassifier()
        selection_model.fit(select_X_train, y_train0)
        # eval model
        select_X_test = selection.transform(x_test)
        y_pred = selection_model.predict(select_X_test)
        predictions = [round(value) for value in y_pred]

        accuracy =  metrics.accuracy_score(y_test,predictions)
        F1 = metrics.f1_score(y_test,predictions)
        recall  = metrics.recall_score(y_test,predictions)
        thresh_list.append(thresh)
        n_list.append(select_X_train.shape[1])
        accuracy_list.append(accuracy)
        F1_list.append(F1)
        recall_list.append(recall)
# save all the paraments
df2 = pd.DataFrame({'thresh':thresh_list,'number':n_list,'accuracy':accuracy_list,'F1':F1_list,'recall':recall})
df2.to_csv('/media/zzh/data/AMP/train_data/classify/Selection_features.csv')