'''
Reference:
https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5?scriptVersionId=61898518&cellId=18
'''

import pandas as pd

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split


from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

df = pd.read_csv('healthcare-dataset-stroke-data.csv')

# Here we will use a Decision Tree to predict the missing BMI
# A really fantastic and intelligent way to deal with blanks, from Thoman Konstantin in:
# https://www.kaggle.com/thomaskonstantin/analyzing-and-modeling-stroke-data

DT_bmi_pipe = Pipeline(steps=[
    ('scale', StandardScaler()),
    ('lr', DecisionTreeRegressor(random_state=42))
])
X = df[['age', 'gender', 'bmi']].copy()
X.gender = X.gender.replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)

Missing = X[X.bmi.isna()]
X = X[~X.bmi.isna()]
Y = X.pop('bmi')
DT_bmi_pipe.fit(X, Y)
predicted_bmi = pd.Series(DT_bmi_pipe.predict(Missing[['age', 'gender']]), index=Missing.index)
df.loc[Missing.index, 'bmi'] = predicted_bmi

# str_only = df[df['stroke'] == 1]
# no_str_only = df[df['stroke'] == 0]
# # Drop single 'Other' gender
# no_str_only = no_str_only[(no_str_only['gender'] != 'Other')]

# Encoding categorical values

df['gender'] = df['gender'].replace({'Male': 0, 'Female': 1, 'Other': -1}).astype(np.uint8)
df['Residence_type'] = df['Residence_type'].replace({'Rural': 0, 'Urban': 1}).astype(np.uint8)
df['work_type'] = df['work_type'].replace(
    {'Private': 0, 'Self-employed': 1, 'Govt_job': 2, 'children': -1, 'Never_worked': -2}).astype(np.uint8)

X = df[['gender', 'age', 'hypertension', 'heart_disease', 'work_type', 'avg_glucose_level', 'bmi']]
y = df['stroke']
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.3, random_state=42)
# print(X_test.head(2))

# Our data is biased, we can fix this with SMOTE

oversample = SMOTE()
X_train_resh, y_train_resh = oversample.fit_resample(X_train, y_train.ravel())

# Models

# Scale our data in pipeline, then split

rf_pipeline = Pipeline(steps=[('scale', StandardScaler()), ('RF', RandomForestClassifier(random_state=42))])
svm_pipeline = Pipeline(steps=[('scale', StandardScaler()), ('SVM', SVC(random_state=42))])
logreg_pipeline = Pipeline(steps=[('scale', StandardScaler()), ('LR', LogisticRegression(random_state=42))])

# X = upsampled_df.iloc[:,:-1] # X_train_resh
# Y = upsampled_df.iloc[:,-1]# y_train_resh

# retain_x = X.sample(100)
# retain_y = Y.loc[X.index]

# X = X.drop(index=retain_x.index)
# Y = Y.drop(index=retain_x.index)

rf_cv = cross_val_score(rf_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1')
svm_cv = cross_val_score(svm_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1')
logreg_cv = cross_val_score(logreg_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1')

print('Mean f1 scores:')
print('Random Forest mean :', cross_val_score(rf_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1').mean())
print('SVM mean :', cross_val_score(svm_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1').mean())
print('Logistic Regression mean :',
      cross_val_score(logreg_pipeline, X_train_resh, y_train_resh, cv=10, scoring='f1').mean())

rf_pipeline.fit(X_train_resh, y_train_resh)
svm_pipeline.fit(X_train_resh, y_train_resh)
logreg_pipeline.fit(X_train_resh, y_train_resh)

# X = df.loc[:,X.columns]
# Y = df.loc[:,'stroke']

rf_pred = rf_pipeline.predict(X_test)
svm_pred = svm_pipeline.predict(X_test)
logreg_pred = logreg_pipeline.predict(X_test)

rf_cm = confusion_matrix(y_test, rf_pred)
svm_cm = confusion_matrix(y_test, svm_pred)
logreg_cm = confusion_matrix(y_test, logreg_pred)

rf_f1 = f1_score(y_test, rf_pred)
svm_f1 = f1_score(y_test, svm_pred)
logreg_f1 = f1_score(y_test, logreg_pred)

# print('Mean f1 scores:')
#
# print('RF mean :', rf_f1)
# print('SVM mean :', svm_f1)
# print('LR mean :', logreg_f1)
#
# print(classification_report(y_test, rf_pred))
#
# print('Accuracy Score: ', accuracy_score(y_test, rf_pred))

n_estimators = [64, 100, 128, 200]
max_features = [2, 3, 5, 7]
bootstrap = [True, False]

param_grid = {'n_estimators': n_estimators,
              'max_features': max_features,
              'bootstrap': bootstrap}
rfc = RandomForestClassifier()
# Let's use those params now

rfc = RandomForestClassifier(max_features=2, n_estimators=100, bootstrap=True)

rfc.fit(X_train_resh, y_train_resh)

rfc_tuned_pred = rfc.predict(X_test)
print(classification_report(y_test, rfc_tuned_pred))

print('Accuracy Score: ', accuracy_score(y_test, rfc_tuned_pred))
print('F1 Score: ', f1_score(y_test, rfc_tuned_pred))
