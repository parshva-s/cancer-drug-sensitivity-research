import pickle
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def save_pickle(data, file_path):
    with open(file_path, 'wb') as file:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        pickle.dump(data, file)


def load_and_prepare_data(file_path: str, drop_columns: list) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True)
    return df


def split_data(df: pd.DataFrame = None, top_features: list = None, target_variable: str = None, X = None, y = None, test_size=0.2):
    
    # check if top_features is provided
    if top_features is not None:
        X = df[top_features]
    elif X is None:
        raise ValueError("Either top_features or X should be provided.")
    
    # check if target_variable is provided
    if target_variable is not None: 
        y = df[target_variable]
    elif y is None:
        raise ValueError("Either target_variable or y should be provided.")
        
    # train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    # validation split (50% of test set)
    X_val, X_test, y_val, y_test = train_test_split(
        X_test, y_test, test_size=0.5, random_state=42)
    
    # scaling the data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test
