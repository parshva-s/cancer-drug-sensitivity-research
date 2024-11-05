# import time
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split, GridSearchCV
# from sklearn.metrics import mean_squared_error, r2_score
# from scipy.stats import pearsonr
# from sklearn.preprocessing import StandardScaler
# from sklearn.svm import SVR


# def load_and_prepare_data(file_path: str, drop_columns: list) -> pd.DataFrame:
#     """
#     Load dataset from a CSV file and drop specified columns.
#     """
#     df = pd.read_csv(file_path)
#     df.drop(columns=drop_columns, inplace=True)
#     return df


# def perform_pearson_correlation(df: pd.DataFrame, target_variable: str, k: int) -> pd.DataFrame:
#     """
#     Perform Pearson correlation for all features in the dataframe and return the top K features.
#     """
#     pearson_correlations = {}
#     for gene in df.columns:
#         if gene != target_variable:
#             correlation, _ = pearsonr(df[gene], df[target_variable])
#             pearson_correlations[gene] = abs(correlation)

#     pearson_df = pd.DataFrame(list(pearson_correlations.items()), columns=[
#                               'Gene', 'Correlation'])
#     pearson_df_sorted = pearson_df.sort_values(
#         by='Correlation', ascending=False)
#     return pearson_df_sorted.head(k)


# def split_data(df: pd.DataFrame, top_features: list, target_variable: str):
#     """
#     Split the data into training and test sets based on the top features and target variable.
#     """
#     X = df[top_features]
#     y = df[target_variable]
#     X_train, X_test, y_train, y_test = train_test_split(
#         X, y, test_size=0.2, random_state=42)
#     scaler = StandardScaler()
#     X_train_scaled = scaler.fit_transform(X_train)
#     X_test_scaled = scaler.transform(X_test)
#     return X_train_scaled, X_test_scaled, y_train, y_test


# def grid_search_random_forest(X_train, y_train):
#     rf_param_grid = {
#         'n_estimators': [50, 100, 200],
#         'max_depth': [None, 10, 20, 30],
#         'min_samples_split': [2, 5, 10]
#     }
#     rf = RandomForestRegressor(random_state=42)
#     grid_search_rf = GridSearchCV(
#         estimator=rf, param_grid=rf_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_search_rf.fit(X_train, y_train)
#     return grid_search_rf


# def grid_search_svr(X_train, y_train):
#     svr_param_grid = {
#         'C': [0.1, 1, 10],
#         'epsilon': [0.01, 0.1, 1],
#         'kernel': ['rbf', 'linear']
#     }
#     svr = SVR()
#     grid_search_svr = GridSearchCV(
#         estimator=svr, param_grid=svr_param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
#     grid_search_svr.fit(X_train, y_train)
#     return grid_search_svr


# def evaluate_model(model, X_test, y_test):
#     """
#     Evaluate the Random Forest model and return the Mean Squared Error and R² score.
#     """
#     y_pred = model.predict(X_test)
#     mse = mean_squared_error(y_test, y_pred)
#     r2 = r2_score(y_test, y_pred)
#     return mse, r2


# def main():
#     target_variable = "LN_IC50"
#     file_path = "data/drug1003Data.csv"

#     # Load and preprocess data
#     df = load_and_prepare_data(
#         file_path, ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME'], target_variable)

#     # Pearson correlation to select top features
#     top_K_pearson_df = perform_pearson_correlation(df, target_variable, k=200)
#     top_features = top_K_pearson_df['Gene'].values

#     # Split the data
#     X_train, X_test, y_train, y_test = split_data(
#         df, top_features, target_variable)

#     # Perform GridSearchCV for Random Forest Regressor
#     start_time = time.time()
#     best_rf_model = grid_search_random_forest(X_train, y_train)
#     end_time = time.time()
#     print("Random Forest Grid Search time:", end_time - start_time)
#     print("Best parameters for Random Forest:", best_rf_model.best_params_)
#     rf_mse, rf_r2 = evaluate_model(best_rf_model, X_test, y_test)
#     print(f"Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}")

#     # Perform GridSearchCV for SVR
#     start_time = time.time()
#     best_svr_model = grid_search_svr(X_train, y_train)
#     end_time = time.time()
#     print("SVR Grid Search time:", end_time - start_time)
#     print("Best parameters for SVR:", best_svr_model.best_params_)
#     svr_mse, svr_r2 = evaluate_model(best_svr_model, X_test, y_test)
#     print(f"SVR - MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}")


# if __name__ == "__main__":
#     main()


# feature_selection.py
import pandas as pd
from scipy.stats import pearsonr


def perform_pearson_correlation(df: pd.DataFrame, target_variable: str, k: int = 200) -> pd.DataFrame:
    pearson_correlations = {}
    for gene in df.columns:
        if gene != target_variable:
            correlation, _ = pearsonr(df[gene], df[target_variable])
            pearson_correlations[gene] = abs(correlation)

    pearson_df = pd.DataFrame(list(pearson_correlations.items()), columns=[
                              'Gene', 'Correlation'])
    pearson_df_sorted = pearson_df.sort_values(
        by='Correlation', ascending=False)
    return pearson_df_sorted.head(k)
