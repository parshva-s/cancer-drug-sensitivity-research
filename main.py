import time
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor.Dataset import Dataset
from feature_reduction.NMF_expression import apply_nmf
from models.models import grid_search_random_forest, grid_search_svr, grid_search_elastic_net, train_neural_network, evaluate_model
from utils.common import split_data
from feature_reduction.feature_selection import perform_pearson_correlation
from feature_reduction.PCA import perform_pca

def main():
    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"
    drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']
    target_variable = "LN_IC50"
    
    reduction_methods = ['pearson', 'nmf', 'pca']

    # feature parameters
    k_values = [10, 25, 50]
    n_components_list = [10, 20, 50]
    pca_components = [10, 20, 30]

    # initialize dataset
    dataset = Dataset(
        dataset_name=dataset_name,
        type=type,
        gene_file=gene_file_name,
        IC50_file=drug_file_name,
        data_directory=data_directory,
    )
    df = dataset.create_data(drug_id)
    df.drop(columns=drop_columns, inplace=True)

    # Collect results
    results = []
    # Test different feature selection and dimensionality reduction techniques
    for method in reduction_methods:
        if method == 'pearson':
            for k in k_values:
                top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
                top_features = top_K_pearson_df['Gene'].values
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    df, top_features, target_variable)
                
                evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results)
        elif method == 'nmf':
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            for n_components in n_components_list:
                W, _, _ = apply_nmf(X, n_components)
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                    X = pd.DataFrame(W), y = y)

                evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, n_components, method, results)
        elif method == 'pca':
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            for n_components in pca_components:
                X_train, X_val, X_test, y_train, y_val, y_test = split_data(X = X, y = y)
                X_train, X_val, X_test = perform_pca(n_components, X_train, X_val, X_test)

                evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, n_components, method, results)

    plot_results(results)

def evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results):
    """
    Evaluate different models for a given feature reduction method.

    Args:
        X_train (pd.DataFrame): data for training
        X_val (pd.DataFrame): data for validation
        X_test (pd.DataFrame): data for testing
        y_train (pd.DataFrame): target for training
        y_val (pd.DataFrame): target for validation
        y_test (pd.DataFrame): target for testing
        k (int): number of features/components
        method (Any): reduction method used
        results (list): list to store results. A tuple of (method, model, k, mse, time) is appended to the list.
    """
    
    # evaluate Random Forest
    start_time = time.time()
    rf_model = grid_search_random_forest(X_train, y_train)
    rf_mse, _ = evaluate_model(rf_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Random Forest', k, rf_mse, elapsed_time))

    # evaluate SVR
    start_time = time.time()
    svr_model = grid_search_svr(X_train, y_train)
    svr_mse, _ = evaluate_model(svr_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'SVR', k, svr_mse, elapsed_time))

    # evaluate Elastic Net
    start_time = time.time()
    en_model = grid_search_elastic_net(X_train, y_train)
    en_mse, _ = evaluate_model(en_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Elastic Net', k, en_mse, elapsed_time))

    # evaluate Neural Network
    start_time = time.time()
    nn_model, _ = train_neural_network(X_train, y_train, X_val, y_val)
    nn_mse, _ = evaluate_model(nn_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Neural Network', k, nn_mse, elapsed_time))

def plot_results(results):
    """
    Plot the results of the evaluation.

    Args:
        results (list): list of results for each model and reduction method
    """
    # convert results to DataFrame
    df_results = pd.DataFrame(results, columns=['Method', 'Model', 'K/Components', 'MSE', 'Time'])

    # generate plots for each feature reduction method
    methods = df_results['Method'].unique()
    for method in methods:
        plt.figure(figsize=(12, 6))
        subset = df_results[df_results['Method'] == method]
        for model in subset['Model'].unique():
            model_subset = subset[subset['Model'] == model]
            plt.plot(model_subset['K/Components'], model_subset['MSE'], marker='o', label=model)

        plt.xlabel('Number of K/Components')
        plt.ylabel('Mean Squared Error')
        plt.title(f'Model Performance for {method} Reduction Technique')
        plt.legend()
        plt.savefig(f'./data/mse_{method}.png')
        plt.close()

if __name__ == "__main__":
    main()
