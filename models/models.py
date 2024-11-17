import time
import keras
from sklearn.ensemble import RandomForestRegressor
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

def train_neural_network(X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
    """
    Training neural network model

    Args:
        X_train (pd.Dataframe): features for training
        y_train (pd.Dataframe): target for training
        X_val (pd.Dataframe): features for validation
        y_val (pd.Dataframe): target for validation
        epochs (int, optional): epochs for training. Defaults to 100.
        batch_size (int, optional): batches for training. Defaults to 32.
    
    Returns:
        model: trained neural network model
        history: training history
    """
    
    # neural network model architecture
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1)  # Output layer for regression
    ])

    model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
