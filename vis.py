import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.gridspec as grid_spec
import seaborn as sns
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_score, f1_score

pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

df = pd.read_csv('healthcare-dataset-stroke-data.csv')
example = df.head(3)
# with open('output.txt', 'w') as f:
#     f.write('first 3 rows: \n' + str(example) + '\n\nsummary: \n' + str(df.describe()) + '\n\nmissing data: \n' + str(
#         df.isnull().sum()
#         ))

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
print('Missing values: ', sum(df.isnull().sum()))
