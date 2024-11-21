# PCA Reduction Technique
import time
import pandas as pd
import numpy as np
import sys
import os
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))
from models.models import evaluate_model, grid_search_random_forest
from utils.common import split_data

def perform_pca(k, X_train, X_val, X_test):
    """
    Applies PCA to X_train with specified amount of components and transforms the rest of the data
    
    Parameters:
    k (int): Number of components
    X_train: Training data after splitting
    X_val: Validation data after splitting
    X_test: Test data after splitting

    Returns:
    X_train, X_val, X_test: All features after transforming with PCA
    """
    pca = PCA(n_components=k)
    X_train = pca.fit_transform(X_train)
    X_train = pd.DataFrame(X_train, columns=[f"PC{i+1}" for i in range(k)])
    X_val = pca.transform(X_val)
    X_val = pd.DataFrame(X_val, columns=[f"PC{i+1}" for i in range(k)])
    X_test = pca.transform(X_test)
    X_test = pd.DataFrame(X_test, columns=[f"PC{i+1}" for i in range(k)])
    return X_train, X_val, X_test

def number_of_components(df):
    """
    Determines the minimum number of principal components required to retain 90% of the variance in the data.

    Parameters:
    df (DataFrame): Scaled Input Features where each row is a sample and each column is a feature.

    Returns:
    tuple: A tuple containing:
        - n_components (int): The number of principal components required to retain 90% of the variance.
        - cumulative_explained_variance (float): The actual cumulative explained variance achieved with the selected number of components.
    """
    pca = PCA()
    pca.fit(df)

    # Calculate the explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Calculate the cumulative explained variance
    cumulative_explained_variance = np.cumsum(explained_variance_ratio)

    # Find the number of components for 95% variance
    n_components = np.argmax(cumulative_explained_variance >= 0.90) + 1

    return (n_components, cumulative_explained_variance[n_components-1])

def main():
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

    k_values = [5, 10, 25, 50]
    mses = []
    target_variable = "LN_IC50"
    top_features = [col for col in df.columns if col != "LN_IC50"]
    
    best_rf_params, best_rf_mse = None, float('inf')
    rf_param_grid = {
        'n_estimators': [10, 25, 50, 100, 200],
        'max_depth': [10],
        'min_samples_split': [2]
    }
    
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(
        df, top_features, target_variable)
    num_comp, _ = number_of_components(X_train)
    print(f"Number of Components required retain 90% of the variance {num_comp}")
    
    for k in k_values:
        # Split the data
        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
            df, top_features, target_variable)

        # GridSearchCV for Random Forest
        start_time = time.time()
        X_train, X_val, X_test = perform_pca(k, X_train, X_val, X_test)
        rf_model = grid_search_random_forest(X_train, y_train)
        rf_mse, rf_r2 = evaluate_model(rf_model, X_test, y_test)
        end_time = time.time()
        
        mses.append(rf_mse)
        print(f"PCA={k}, Random Forest - MSE: {rf_mse:.4f}, R²: {rf_r2:.4f}, Time: {end_time - start_time:.2f}s")
        
        # Update best model if this one has a lower MSE
        if rf_mse < best_rf_mse:
            best_rf_mse = rf_mse
            best_rf_k = k
            best_rf_params = rf_model.best_params_
        
    print(f"Best RF Parameters: {best_rf_params}")
    
    plt.figure(figsize=(8, 5))
    plt.plot(k_values, mses, marker='o')
    plt.xlabel('Number of Principal Components (k)')
    plt.ylabel('Mean Squared Error (MSE)')
    plt.title('MSE vs Number of Principal Components')
    plt.savefig('./data/pca_mse_vs_k_plot.png')
    plt.close()
    
if __name__ == "__main__":
    main()