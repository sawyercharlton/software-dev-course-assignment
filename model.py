'''
Reference:
https://www.kaggle.com/code/onurrr90/heart-science-stroke-prediction-with-ai
'''

import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import KFold, cross_val_score, GridSearchCV
from catboost import CatBoostClassifier


pd.set_option('display.max_columns', None)
pd.set_option('expand_frame_repr', False)

train = pd.read_csv("./healthcare-dataset-stroke-data.csv")
train = train.drop(columns=['id'])
# Apply encoding to categorical variables and keep numerical variables unchanged
processed_data = train.apply(lambda column: LabelEncoder().fit_transform(column) if column.dtype == 'O' else column)

X_train = processed_data.drop(['stroke'], axis=1)
Y_train = processed_data['stroke']
X_train = pd.DataFrame(StandardScaler().fit_transform(X_train), columns=X_train.columns)

print(X_train)


def optimize_hyperparameters(model, param_grid, X_train, Y_train, cv_splits=5, random_state=1, verbose=1, n_jobs=-1):
    # Prepare the cross-validation procedure
    kf = KFold(n_splits=cv_splits, shuffle=True, random_state=random_state)

    # Initialize GridSearchCV
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=kf, verbose=verbose,
                               n_jobs=n_jobs)

    # Fit GridSearchCV
    grid_search.fit(X_train, Y_train)

    # Return the best parameters and the best score
    return grid_search.best_params_, grid_search.best_score_


def run_model(model, X_train, Y_train):
    # Prepare the cross-validation procedure
    kf = KFold(n_splits=5, shuffle=True, random_state=1)

    # Evaluate the model
    accuracy = cross_val_score(model, X_train, Y_train, cv=kf, scoring='accuracy')

    # Calculate the average accuracy
    average_accuracy = accuracy.mean()

    # Print the results
    print("Accuracy for each fold:", accuracy)
    print("Average Accuracy across all folds:", average_accuracy)


# catboost_param_grid = {
#     'iterations': [100, 200],
#     'depth': [3, 4, 5, 6, 7],
#     'learning_rate': [0.1, 0.2, 0.3],
#     'loss_function': ['Logloss'],
# }
#
# catboost_model = CatBoostClassifier(eval_metric='Accuracy', random_seed=1, verbose=0)
# best_params_catboost, best_score_catboost = optimize_hyperparameters(catboost_model, catboost_param_grid, X_train, Y_train)

# print("--------------------------------------------------------------------") # use as separator
# print("CatBoost Best Parameters:", best_params_catboost)
# print("CatBoost Best Accuracy:", best_score_catboost)
# print("--------------------------------------------------------------------") # use as separator


cat_boost = CatBoostClassifier(
    iterations=100,
    depth=4,
    learning_rate=0.1,
    loss_function='Logloss',
    eval_metric='Accuracy',
    random_seed=1,
    metric_period=100,
    verbose=False
)

cat_boost.fit(X_train, Y_train, verbose=False)
cat_boost.save_model("catboost")
