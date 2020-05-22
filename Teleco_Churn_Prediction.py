# -*- coding: utf-8 -*-
"""
Created on Mon May 18 14:31:48 2020

@author: myy
"""
import pandas as pd
import numpy as np
import warnings
import seaborn as sns
warnings.filterwarnings("ignore")

filename = r'C:/Users/myy/Desktop/电信用户流失/WA_Fn-UseC_-Telco-Customer-Churn.csv'
data = pd.read_csv(filename)
data.info()

# 查看数据具体模样
for item in data.columns:
    print(item)
    print (data[item].unique())

data['Churn'].value_counts(sort = False)
data.drop(['customerID'],axis=1,inplace=True)
#data['gender'] = data['gender'].replace('male',0)
#data['gender'] = data['gender'].replace('female',1)
data['gender'] = data['gender'].map(lambda s :1  if s =='Yes' else 0)
data['Churn'] = data['Churn'].map(lambda s :1  if s =='Yes' else 0)
data['Partner'] = data['Partner'].map(lambda s :1  if s =='Yes' else 0)
data['Dependents'] = data['Dependents'].map(lambda s :1  if s =='Yes' else 0)
data['PhoneService'] = data['PhoneService'].map(lambda s :1  if s =='Yes' else 0)
data['Dependents'] = data['Dependents'].map(lambda s :1  if s =='Yes' else 0)
data['MultipleLines'] = data['MultipleLines'].replace('No phone service',0)
data['MultipleLines'] = data['MultipleLines'].map(lambda s :1  if s =='Yes' else 0)
data['MultipleLines'].value_counts()

data['OnlineSecurity'] = data['OnlineSecurity'].replace('No phone service',0)
data['OnlineSecurity'] = data['OnlineSecurity'].map(lambda s :1  if s =='Yes' else 0)

data['OnlineBackup'] = data['OnlineBackup'].replace('No phone service',0)
data['OnlineBackup'] = data['OnlineBackup'].map(lambda s :1  if s =='Yes' else 0)

data['DeviceProtection'] = data['DeviceProtection'].replace('No phone service',0)
data['DeviceProtection'] = data['DeviceProtection'].map(lambda s :1  if s =='Yes' else 0)

data['TechSupport'] = data['TechSupport'].replace('No phone service',0)
data['TechSupport'] = data['TechSupport'].map(lambda s :1  if s =='Yes' else 0)

data['StreamingTV'] = data['StreamingTV'].replace('No phone service',0)
data['StreamingTV'] = data['StreamingTV'].map(lambda s :1  if s =='Yes' else 0)

data['StreamingMovies'] = data['StreamingMovies'].replace('No phone service',0)
data['StreamingMovies'] = data['StreamingMovies'].map(lambda s :1  if s =='Yes' else 0)

data['Has_InternetService'] = data['InternetService'].map(lambda s :0  if s =='No' else 1)
data['Fiber_optic'] = data['InternetService'].map(lambda s :1  if s =='Fiber optic' else 0)
data['DSL'] = data['InternetService'].map(lambda s :1  if s =='DSL' else 0)
data.drop(['InternetService'], axis=1, inplace=True)
data['PaperlessBilling'] = data['PaperlessBilling'].map(lambda s :1  if s =='Yes' else 0)

data = pd.get_dummies(data=data, columns=['PaymentMethod'])

data = pd.get_dummies(data=data, columns=['Contract'])
g = sns.factorplot(x="Churn", y = "MonthlyCharges",data = data, kind="box", palette = "Pastel1")

# 有空值
data = data[data['TotalCharges'] != " "]

y = data['Churn']
X = data.drop(labels = ['Churn'],axis = 1)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report

X_train,X_test,y_train,y_test = train_test_split(X,y,random_state=2,test_size=0.2)
RFC = RandomForestClassifier(class_weight={1: 1.5}, max_depth=3, max_features='log2', n_estimators=500)
MD = RFC.fit(X_train,y_train)

# 特征重要性
Rfclf_fea = pd.DataFrame(MD.feature_importances_)
Rfclf_fea["Feature"] = list(X_train) 
Rfclf_fea.sort_values(by=0, ascending=False)

# 训练集混淆矩阵
y_pred = MD.predict(X_train)
print(confusion_matrix(y_train, y_pred))
print(classification_report(y_train, y_pred))
# 验证集混淆矩阵
y_pred1 = MD.predict(X_test)
print(confusion_matrix(y_test, y_pred1))
print(classification_report(y_test, y_pred1))

clf_score = cross_val_score(RFC, X_train, y_train, cv=10)
print(clf_score)
clf_score.mean()


# 网格搜索
param_grid  = { 
                'n_estimators' : [500,1200],
               # 'min_samples_split': [2,5,10,15,100],
               # 'min_samples_leaf': [1,2,5,10],
                'max_depth': range(1,5,2),
                'max_features' : ('log2', 'sqrt'),
                'class_weight':[{1: w} for w in [1,1.5]]
              }
GridRF = GridSearchCV(RandomForestClassifier(random_state=15), param_grid)
GridRF.fit(X, y)
#RF_preds = GridRF.predict_proba(X_test)[:, 1]
#RF_performance = roc_auc_score(Y_test, RF_preds)
print(
    #'DecisionTree: Area under the ROC curve = {}'.format(RF_performance)
     "\nBest parameters \n" + str(GridRF.best_params_))


# auc曲线
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
test_est_p = RFC.predict_proba(X_test)[:,1]
train_est_p = RFC.predict_proba(X_train)[:,1]

fpr_test,tpr_test,th_test = metrics.roc_curve(y_test,test_est_p)
fpr_train,tpr_train,th_train = metrics.roc_curve(y_train,train_est_p)

plt.figure(figsize=[9,9])
plt.plot(fpr_test,tpr_test,'r--')
plt.plot(fpr_train,tpr_train,'g-')

print('AUC = %.4f' %metrics.auc(fpr_test,tpr_test))
print('ROC = %.4f' %metrics.roc_auc_score(y_test,test_est_p))