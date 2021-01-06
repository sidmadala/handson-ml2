import sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR  # regressor


# Ensure that param grid follows inputs for model
param_grid = [
        {"kernel": ['linear'], "C": [1.0, 3.0, 10., 30.]},
        {"kernel": ['rbf'], "C": [1.0, 3.0, 10., 30.], "gamma": [0.01, 0.03, 0.1, 0.3]}
    ]


# Initializing SVM
svm_reg = SVR()

# Inputting SVM into grid search and fitting
grid_search = GridSearchCV(estimator=svm_reg, param_grid=param_grid, cv=5, scoring="neg_mean_squared_error", verbose=2)
# grid_search.fit(housing_prepared, housing_labels)
