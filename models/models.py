import time
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV


def grid_search_random_forest(X_train, y_train, rf_param_grid=None):
    """
    Perform GridSearchCV for Random Forest Regressor.
    If no rf_param_grid is provided, default grid parameters will be used.
    """
    if rf_param_grid is None:
        rf_param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10]
        }
    rf = RandomForestRegressor(random_state=42)
    grid_search_rf = GridSearchCV(
        estimator=rf, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_rf.fit(X_train, y_train)
    return grid_search_rf


def grid_search_svr(X_train, y_train, svr_param_grid=None):
    """
    Perform GridSearchCV for Support Vector Regressor (SVR).
    If no svr_param_grid is provided, default grid parameters will be used.
    """
    if svr_param_grid is None:
        svr_param_grid = {
            'C': [0.1, 1, 10],
            'epsilon': [0.01, 0.1, 1],
            'kernel': ['rbf', 'linear']
        }
    svr = SVR()
    grid_search_svr = GridSearchCV(
        estimator=svr, param_grid=svr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_svr.fit(X_train, y_train)
    return grid_search_svr

def grid_search_elastic_net(X_train, y_train, en_param_grid=None):
    """
    Perform GridSearchCV for Elastic Net Regressor.
    If no en_param_grid is provided, default grid parameters will be used.
    """
    if en_param_grid is None:
        en_param_grid = {
            'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
            'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
        }
    en = ElasticNet(random_state=42)
    grid_search_en = GridSearchCV(
        estimator=en, param_grid=en_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    grid_search_en.fit(X_train, y_train)
    return grid_search_en


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
