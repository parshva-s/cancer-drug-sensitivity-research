import time
from scipy.stats import pearsonr
import pandas as pd
import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from models.models import evaluate_model, grid_search_random_forest, grid_search_svr, train_neural_network
from utils.common import split_data

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


if __name__ == "__main__":
    from preprocessor.Dataset import Dataset
    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"
    drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']

    # create dataset to be used
    dataset = Dataset(
        dataset_name,
        type,
        gene_file_name,
        drug_file_name,
        data_directory)

    df = dataset.create_data(drug_id)
    df.drop(columns=drop_columns, inplace=True)

    k_values = [50]
    rf_param_grid = {
        'n_estimators': [50],
        'max_depth': [10],
        'min_samples_split': [2]
    }
    svr_param_grid = {
        'C': [0.1],
        'epsilon': [1],
        'kernel': ['rbf']
    }
    target_variable = "LN_IC50"

    best_rf_mse, best_svr_mse = float('inf'), float('inf')
    best_rf_k, best_svr_k = None, None
    best_rf_params, best_svr_params = None, None
    best_nn_k, best_nn_mse, best_nn_params = None, float('inf'), None

    for k in k_values:
        top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
        top_features = top_K_pearson_df['Gene'].values

        # Split the data
        X_train, X_test, X_val, y_val, y_train, y_test = split_data(
            df, top_features, target_variable)

        # GridSearchCV for Random Forest
        start_time = time.time()
        rf_model = grid_search_random_forest(X_train, y_train, rf_param_grid)
        rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
        end_time = time.time()
        print(
            f"K={k}, Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}, Time: {end_time - start_time:.2f}s")

        # Update best model if this one has a lower MSE
        if rf_mse < best_rf_mse:
            best_rf_mse = rf_mse
            best_rf_k = k
            best_rf_params = rf_model.best_params_

        # GridSearchCV for SVR
        start_time = time.time()
        svr_model = grid_search_svr(X_train, y_train, svr_param_grid)
        svr_mse, svr_r2 = evaluate_model(svr_model, X_test, y_test)
        end_time = time.time()
        print(
            f"K={k}, SVR - MSE: {svr_mse:.4f}, R²: {svr_r2:.4f}, Time: {end_time - start_time:.2f}s")
        
        # Update best model if this one has a lower MSE
        if svr_mse < best_svr_mse:
            best_svr_mse = svr_mse
            best_svr_k = k
            best_svr_params = svr_model.best_params_        
        
        # Using Neural Network
        start_time = time.time()
        nn_model, history = train_neural_network(X_train, y_train, X_val, y_val)
        nn_mse, nn_r2 = evaluate_model(nn_model, X_test, y_test)
        end_time = time.time()
        print(
            f"K={k}, Neural Network - MSE: {nn_mse:.4f}, R²: {nn_r2:.4f}, Time: {end_time - start_time:.2f}s")
        
        # Update best model if this one has a lower MSE
        if nn_mse < best_nn_mse:
            best_nn_mse = nn_mse
            best_nn_k = k
            best_nn_params = nn_model.get_config()

    # Print the best results
    print("\nBest Random Forest Model:")
    print(f"K={best_rf_k}, MSE={best_rf_mse:.4f}, Params={best_rf_params}")

    print("\nBest SVR Model:")
    print(f"K={best_svr_k}, MSE={best_svr_mse:.4f}, Params={best_svr_params}")
