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


def split_data(df: pd.DataFrame, top_features: list, target_variable: str, test_size=0.2):
    X = df[top_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test
