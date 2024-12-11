import os
import pickle

import keras
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVR
from torch_geometric.data import DataLoader

from models.gnn import create_graph_data, GNNModel

models_directory = "data/saved_models/"


def grid_search_random_forest(drug_id, method, k, X_train, y_train, rf_param_grid=None):
    """
    Perform GridSearchCV for Random Forest Regressor.
    If no rf_param_grid is provided, default grid parameters will be used.
    """
    rf_model_path = f'{models_directory}{drug_id}_{method}_{k}_rf.pkl'
    grid_search_rf = None
    if os.path.exists(rf_model_path):
        print(f"Model file '{rf_model_path}' found. Loading the model...")
        with open(rf_model_path, 'rb') as file:
            grid_search_rf = pickle.load(file)
    else:
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

        with open(rf_model_path, 'wb') as file:
            pickle.dump(grid_search_rf, file)
        print(f"Model trained and saved to '{rf_model_path}'.")
    return grid_search_rf


def grid_search_svr(drug_id, method, k, X_train, y_train, svr_param_grid=None):
    """
    Perform GridSearchCV for Support Vector Regressor (SVR).
    If no svr_param_grid is provided, default grid parameters will be used.
    """
    svr_model_path = f'{models_directory}{drug_id}_{method}_{k}_svr.pkl'
    grid_search_svr = None
    if os.path.exists(svr_model_path):
        print(f"Model file '{svr_model_path}' found. Loading the model...")
        with open(svr_model_path, 'rb') as file:
            grid_search_svr = pickle.load(file)
    else:
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

        with open(svr_model_path, 'wb') as file:
            pickle.dump(grid_search_svr, file)
        print(f"Model trained and saved to '{svr_model_path}'.")
    return grid_search_svr


def grid_search_elastic_net(drug_id, method, k, X_train, y_train, en_param_grid=None):
    """
    Perform GridSearchCV for Elastic Net Regressor.
    If no en_param_grid is provided, default grid parameters will be used.
    """
    en_model_path = f'{models_directory}{drug_id}_{method}_{k}_en.pkl'
    grid_search_en = None
    if os.path.exists(en_model_path):
        print(f"Model file '{en_model_path}' found. Loading the model...")
        with open(en_model_path, 'rb') as file:
            grid_search_en = pickle.load(file)
    else:
        if en_param_grid is None:
            en_param_grid = {
                'alpha': [0.1, 0.5, 1.0, 5.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]
            }
        en = ElasticNet(random_state=42)
        grid_search_en = GridSearchCV(
            estimator=en, param_grid=en_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
        grid_search_en.fit(X_train, y_train)

        with open(en_model_path, 'wb') as file:
            pickle.dump(grid_search_en, file)
        print(f"Model trained and saved to '{en_model_path}'.")
    return grid_search_en


def train_neural_network(drug_id, method, k, X_train, y_train, X_val, y_val, epochs=100, batch_size=32):
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

    nn_model_path = f'{models_directory}{drug_id}_{method}_{k}_nn.h5'
    model, history = None, None
    if os.path.exists(nn_model_path):
        print(f"Model file '{nn_model_path}' found. Loading the model...")
        model = keras.models.load_model(nn_model_path)
    else:
        # neural network model architecture
        model = keras.Sequential([
            keras.layers.Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(1)  # Output layer for regression
        ])

        model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mae'])

        # training the model
        history = model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=epochs,
            batch_size=batch_size,
            verbose=0
        )
        model.save(nn_model_path)
        print(f"Model trained and saved to '{nn_model_path}'.")
    return model, history


def train_gnn_model(drug_id, method, k, X_train, y_train, X_val, y_val, hidden_dim=64, epochs=25):
    """
    Train or load a GNN model for a specific drug ID and return the trained model.

    Args:
        drug_id (str): Drug ID for saving/loading the model.
        method (str): Method name for model identification.
        k (int): Some parameter for model identification (e.g., K genes).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        input_dim (int): Number of input features for each node.
        epochs (int, optional): Number of epochs to train the model. Defaults to 25.

    Returns:
        model: Trained GNN model.
    """
    train_data = create_graph_data(X_train, y_train.values)
    val_data = create_graph_data(X_val, y_val.values)

    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=32, shuffle=False)

    model_path = f"{models_directory}{drug_id}_{method}_{k}_gnn.pth"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = GNNModel(1, hidden_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.MSELoss()

    # Check if the model already exists
    if os.path.exists(model_path):
        print(f"Model file '{model_path}' found. Loading the model...")
        model.load_state_dict(torch.load(model_path, map_location=device))
    else:
        print(f"Training new model for drug_id: {drug_id}, method: {method}, k: {k}")
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for data in train_loader:
                data = data.to(device)
                optimizer.zero_grad()
                output = model(data).squeeze()
                loss = criterion(output, data.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validation phase, RN validation data is not used for anything
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for data in val_loader:
                    data = data.to(device)
                    output = model(data).squeeze()
                    loss = criterion(output, data.y)
                    val_loss += loss.item()

            print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model trained and saved to '{model_path}'.")

    return model


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2
