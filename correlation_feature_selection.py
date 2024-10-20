import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def load_and_prepare_data(file_path: str, drop_columns: list, target_variable: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and drop specified columns.
    """
    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True)
    return df


def perform_pearson_correlation(df: pd.DataFrame, target_variable: str, k: int = 200) -> pd.DataFrame:
    """
    Perform Pearson correlation for all features in the dataframe and return the top K features.
    """
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


def split_data(df: pd.DataFrame, top_features: list, target_variable: str):
    """
    Split the data into training and test sets based on the top features and target variable.
    """
    X = df[top_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators: int = 100):
    """
    Train a Random Forest regressor with the given training data.
    """
    rf_regressor = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor


def evaluate_model(rf_regressor, X_test, y_test):
    """
    Evaluate the Random Forest model and return the Mean Squared Error and R² score.
    """
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def print_feature_importances(rf_regressor, top_features: list):
    """
    Print the feature importances from the Random Forest model.
    """
    importances = rf_regressor.feature_importances_
    for i, feature in enumerate(top_features):
        print(f"Feature: {feature}, Importance: {importances[i]:.4f}")


def grid_search_pearson_random_forest_for_plot(file_path: str, target_variable: str, k_values: list, n_estimators_values: list):
    """
    Perform grid search to find MSE for each combination of k (top features) and n_estimators (trees in RF).
    Store MSE values and plot them in a 3D graph.
    """
    df = load_and_prepare_data(
        file_path, ['AUC', 'Z_SCORE', 'RMSE'], target_variable)

    mse_values = np.zeros((len(k_values), len(n_estimators_values)))

    for i, k in enumerate(k_values):
        for j, n_estimators in enumerate(n_estimators_values):
            top_K_pearson_df = perform_pearson_correlation(
                df, target_variable, k)
            top_features = top_K_pearson_df['Gene'].values

            X_train, X_test, y_train, y_test = split_data(
                df, top_features, target_variable)

            rf_regressor = train_random_forest(X_train, y_train, n_estimators)

            mse, r2 = evaluate_model(rf_regressor, X_test, y_test)

            mse_values[i, j] = mse

    return mse_values, k_values, n_estimators_values


def plot_3d_mse(k_values, n_estimators_values, mse_values):
    """
    Plot the MSE values as a 3D surface plot with k and n_estimators on the axes.
    """
    K, N_estimators = np.meshgrid(k_values, n_estimators_values)
    mse_values = mse_values.T

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Create the surface plot
    ax.plot_surface(K, N_estimators, mse_values, cmap='viridis')

    ax.set_xlabel('Number of Top Features (k)')
    ax.set_ylabel('Number of Estimators (n_estimators)')
    ax.set_zlabel('Mean Squared Error (MSE)')
    ax.set_title('MSE vs. k and n_estimators')

    plt.show()


k_values = [5, 10, 20, 50]
n_estimators_values = [50, 100, 200, 500]

start_time = time.time()
# Perform grid search and get MSE values
mse_values, k_values, n_estimators_values = grid_search_pearson_random_forest_for_plot(
    "data/final_dataset.csv", "LN_IC50", k_values, n_estimators_values)
end_time = time.time()

print(f"Time taken: {end_time - start_time:.2f} seconds")

# Plot the MSE values in a 3D graph
plot_3d_mse(k_values, n_estimators_values, mse_values)
