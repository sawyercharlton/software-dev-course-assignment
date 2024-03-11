'''
Reference:
https://www.kaggle.com/code/joshuaswords/predicting-a-stroke-shap-lime-explainer-eli5?scriptVersionId=61898518&cellId=18
'''


import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
# with open('data_summary.txt', 'w') as f:
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
# print('Missing values: ', sum(df.isnull().sum()))

str_only = df[df['stroke'] == 1]
no_str_only = df[df['stroke'] == 0]
# Drop single 'Other' gender
no_str_only = no_str_only[(no_str_only['gender'] != 'Other')]

fig = plt.figure(figsize=(22, 15))
gs = fig.add_gridspec(3, 3)
gs.update(wspace=0.35, hspace=0.27)
ax0 = fig.add_subplot(gs[0, 0])
ax1 = fig.add_subplot(gs[0, 1])
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[1, 2])
ax6 = fig.add_subplot(gs[2, 0])
ax7 = fig.add_subplot(gs[2, 1])
ax8 = fig.add_subplot(gs[2, 2])

background_color = "#f6f6f6"
fig.patch.set_facecolor(background_color)  # figure background color

# Plots

## Age


ax0.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
positive = pd.DataFrame(str_only["age"])
negative = pd.DataFrame(no_str_only["age"])
sns.kdeplot(positive["age"], ax=ax0, color="#810f13", fill=True, ec='black', label="positive")
sns.kdeplot(negative["age"], ax=ax0, color="#9bb7d4", fill=True, ec='black', label="negative")
# ax3.text(0.29, 13, 'Age',
#        fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax0.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax0.set_ylabel('')
ax0.set_xlabel('')
ax0.text(-20, 0.0465, 'Age', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")

# Smoking
positive = pd.DataFrame(str_only["smoking_status"].value_counts())
positive["Percentage"] = positive["smoking_status"].apply(lambda x: x / sum(positive["smoking_status"]) * 100)
negative = pd.DataFrame(no_str_only["smoking_status"].value_counts())
negative["Percentage"] = negative["smoking_status"].apply(lambda x: x / sum(negative["smoking_status"]) * 100)

ax1.text(0, 4, 'Smoking Status', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax1.barh(positive.index, positive['Percentage'], color="#810f13", zorder=3, height=0.7)
ax1.barh(negative.index, negative['Percentage'], color="#9bb7d4", zorder=3, ec='black', height=0.3)
ax1.xaxis.set_major_formatter(mtick.PercentFormatter())
ax1.xaxis.set_major_locator(mtick.MultipleLocator(10))

##
# Ax2 - GENDER
positive = pd.DataFrame(str_only["gender"].value_counts())
positive["Percentage"] = positive["gender"].apply(lambda x: x / sum(positive["gender"]) * 100)
negative = pd.DataFrame(no_str_only["gender"].value_counts())
negative["Percentage"] = negative["gender"].apply(lambda x: x / sum(negative["gender"]) * 100)

x = np.arange(len(positive))
ax2.text(-0.4, 68.5, 'Gender', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax2.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
ax2.bar(x, height=positive["Percentage"], zorder=3, color="#810f13", width=0.4)
ax2.bar(x + 0.4, height=negative["Percentage"], zorder=3, color="#9bb7d4", width=0.4)
ax2.set_xticks(x + 0.4 / 2)
ax2.set_xticklabels(['Male', 'Female'])
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())
ax2.yaxis.set_major_locator(mtick.MultipleLocator(10))
for i, j in zip([0, 1], positive["Percentage"]):
    ax2.annotate(f'{j:0.0f}%', xy=(i, j / 2), color='#f6f6f6', horizontalalignment='center', verticalalignment='center')
for i, j in zip([0, 1], negative["Percentage"]):
    ax2.annotate(f'{j:0.0f}%', xy=(i + 0.4, j / 2), color='#f6f6f6', horizontalalignment='center',
                 verticalalignment='center')

# Heart Dis

positive = pd.DataFrame(str_only["heart_disease"].value_counts())
positive["Percentage"] = positive["heart_disease"].apply(lambda x: x / sum(positive["heart_disease"]) * 100)
negative = pd.DataFrame(no_str_only["heart_disease"].value_counts())
negative["Percentage"] = negative["heart_disease"].apply(lambda x: x / sum(negative["heart_disease"]) * 100)

x = np.arange(len(positive))
ax3.text(-0.3, 110, 'Heart Disease', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax3.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
ax3.bar(x, height=positive["Percentage"], zorder=3, color="#810f13", width=0.4)
ax3.bar(x + 0.4, height=negative["Percentage"], zorder=3, color="#9bb7d4", width=0.4)
ax3.set_xticks(x + 0.4 / 2)
ax3.set_xticklabels(['No History', 'History'])
ax3.yaxis.set_major_formatter(mtick.PercentFormatter())
ax3.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i, j in zip([0, 1], positive["Percentage"]):
    ax3.annotate(f'{j:0.0f}%', xy=(i, j / 2), color='#f6f6f6', horizontalalignment='center', verticalalignment='center')
for i, j in zip([0, 1], negative["Percentage"]):
    ax3.annotate(f'{j:0.0f}%', xy=(i + 0.4, j / 2), color='#f6f6f6', horizontalalignment='center',
                 verticalalignment='center')

## AX4 - TITLE

ax4.spines["bottom"].set_visible(False)
ax4.tick_params(left=False, bottom=False)
ax4.set_xticklabels([])
ax4.set_yticklabels([])
ax4.text(0.5, 0.6, 'How much each factor affects the results?\n\n Age is the biggest factor.', horizontalalignment='center',
         verticalalignment='center',
         fontsize=22, fontweight='bold', fontfamily='serif', color="#323232")

ax4.text(0.15, 0.57, "Stroke", fontweight="bold", fontfamily='serif', fontsize=22, color='#810f13')
ax4.text(0.41, 0.57, "&", fontweight="bold", fontfamily='serif', fontsize=22, color='#323232')
ax4.text(0.49, 0.57, "No-Stroke", fontweight="bold", fontfamily='serif', fontsize=22, color='#9bb7d4')

# Glucose

ax5.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
positive = pd.DataFrame(str_only["avg_glucose_level"])
negative = pd.DataFrame(no_str_only["avg_glucose_level"])
sns.kdeplot(positive["avg_glucose_level"], ax=ax5, color="#810f13", ec='black', fill=True, label="positive")
sns.kdeplot(negative["avg_glucose_level"], ax=ax5, color="#9bb7d4", ec='black', fill=True, label="negative")
ax5.text(-55, 0.01855, 'Avg. Glucose Level',
         fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax5.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax5.set_ylabel('')
ax5.set_xlabel('')

## BMI


ax6.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
positive = pd.DataFrame(str_only["bmi"])
negative = pd.DataFrame(no_str_only["bmi"])
sns.kdeplot(positive["bmi"], ax=ax6, color="#810f13", ec='black', fill=True, label="positive")
sns.kdeplot(negative["bmi"], ax=ax6, color="#9bb7d4", ec='black', fill=True, label="negative")
ax6.text(-0.06, 0.09, 'BMI',
         fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax6.yaxis.set_major_locator(mtick.MultipleLocator(2))
ax6.set_ylabel('')
ax6.set_xlabel('')

# Work Type

positive = pd.DataFrame(str_only["work_type"].value_counts())
positive["Percentage"] = positive["work_type"].apply(lambda x: x / sum(positive["work_type"]) * 100)
positive = positive.sort_index()

negative = pd.DataFrame(no_str_only["work_type"].value_counts())
negative["Percentage"] = negative["work_type"].apply(lambda x: x / sum(negative["work_type"]) * 100)
negative = negative.sort_index()

ax7.bar(negative.index, height=negative["Percentage"], zorder=3, color="#9bb7d4", width=0.05)
ax7.scatter(negative.index, negative["Percentage"], zorder=3, s=200, color="#9bb7d4")
ax7.bar(np.arange(len(positive.index)) + 0.4, height=positive["Percentage"], zorder=3, color="#810f13", width=0.05)
ax7.scatter(np.arange(len(positive.index)) + 0.4, positive["Percentage"], zorder=3, s=200, color="#810f13")

ax7.yaxis.set_major_formatter(mtick.PercentFormatter())
ax7.yaxis.set_major_locator(mtick.MultipleLocator(10))
ax7.set_xticks(np.arange(len(positive.index)) + 0.4 / 2)
ax7.set_xticklabels(list(positive.index), rotation=0)
ax7.text(-0.5, 66, 'Work Type', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")

# hypertension

positive = pd.DataFrame(str_only["hypertension"].value_counts())
positive["Percentage"] = positive["hypertension"].apply(lambda x: x / sum(positive["hypertension"]) * 100)
negative = pd.DataFrame(no_str_only["hypertension"].value_counts())
negative["Percentage"] = negative["hypertension"].apply(lambda x: x / sum(negative["hypertension"]) * 100)

x = np.arange(len(positive))
ax8.text(-0.45, 100, 'Hypertension', fontsize=14, fontweight='bold', fontfamily='serif', color="#323232")
ax8.grid(color='gray', linestyle=':', axis='y', zorder=0, dashes=(1, 5))
ax8.bar(x, height=positive["Percentage"], zorder=3, color="#810f13", width=0.4)
ax8.bar(x + 0.4, height=negative["Percentage"], zorder=3, color="#9bb7d4", width=0.4)
ax8.set_xticks(x + 0.4 / 2)
ax8.set_xticklabels(['No History', 'History'])
ax8.yaxis.set_major_formatter(mtick.PercentFormatter())
ax8.yaxis.set_major_locator(mtick.MultipleLocator(20))
for i, j in zip([0, 1], positive["Percentage"]):
    ax8.annotate(f'{j:0.0f}%', xy=(i, j / 2), color='#f6f6f6', horizontalalignment='center', verticalalignment='center')
for i, j in zip([0, 1], negative["Percentage"]):
    ax8.annotate(f'{j:0.0f}%', xy=(i + 0.4, j / 2), color='#f6f6f6', horizontalalignment='center',
                 verticalalignment='center')

# tidy up


for s in ["top", "right", "left"]:
    for i in range(0, 9):
        locals()["ax" + str(i)].spines[s].set_visible(False)

for i in range(0, 9):
    locals()["ax" + str(i)].set_facecolor(background_color)
    locals()["ax" + str(i)].tick_params(axis=u'both', which=u'both', length=0)
    locals()["ax" + str(i)].set_facecolor(background_color)

plt.show()