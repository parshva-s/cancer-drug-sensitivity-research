import time
from utils.common import load_and_prepare_data, split_data
from dataReductions.feature_selection import perform_pearson_correlation
from models.models import grid_search_random_forest, grid_search_svr, evaluate_model


def main():
    k_values = [50, 100, 150, 200]
    rf_param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [None, 10, 20, 30],
        'min_samples_split': [2, 5, 10]
    }
    svr_param_grid = {
        'C': [0.1, 1, 10],
        'epsilon': [0.01, 0.1, 1],
        'kernel': ['rbf', 'linear']
    }
    file_path = "data/drug1003Data.csv"
    target_variable = "LN_IC50"

    best_rf_mse, best_svr_mse = float('inf'), float('inf')
    best_rf_k, best_svr_k = None, None
    best_rf_params, best_svr_params = None, None

    df = load_and_prepare_data(
        file_path, ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME'])

    for k in k_values:
        # Perform Pearson correlation to select top features
        top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
        top_features = top_K_pearson_df['Gene'].values

        # Split the data
        X_train, X_test, y_train, y_test = split_data(
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

    # Print the best results
    print("\nBest Random Forest Model:")
    print(f"K={best_rf_k}, MSE={best_rf_mse:.4f}, Params={best_rf_params}")

    print("\nBest SVR Model:")
    print(f"K={best_svr_k}, MSE={best_svr_mse:.4f}, Params={best_svr_params}")


if __name__ == "__main__":
    main()
