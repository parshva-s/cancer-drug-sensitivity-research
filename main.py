import time
import pandas as pd
import matplotlib.pyplot as plt
from preprocessor.Dataset import Dataset
from feature_reduction.NMF_expression import apply_nmf
from models.models import grid_search_random_forest, grid_search_svr, grid_search_elastic_net, train_neural_network, evaluate_model
from utils.common import split_data
from feature_reduction.feature_selection import perform_pearson_correlation
from feature_reduction.PCA import perform_pca
import os

dataset_name = "GDSC2"
type = "expression"
data_directory = "data/"
results_directory = "data/results/"
gene_file_name = "cell_line_expressions.csv"
drug_file_name = "drug_cell_line.csv"
drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']
target_variable = "LN_IC50"
reduction_methods = ['pearson', 'pca']
# feature parameters
k_values = [5, 10, 25, 50, 100, 200, 400, 600]
n_components_list = [5, 10, 25, 50, 100]
pca_components = [5, 10, 25, 50, 100]


def main():
    # initialize dataset
    dataset = Dataset(
        dataset_name=dataset_name,
        type=type,
        gene_file=gene_file_name,
        IC50_file=drug_file_name,
        data_directory=data_directory,
    )
    
    results = []
    dataset.create_drug_id_list()
    drug_ids = dataset.get_drug_id_list()
    
    # FOR TESTING PURPOSES
    # drug_ids = drug_ids[0:50]

    for drug_id in drug_ids:
        df = dataset.create_data(drug_id)
        df.drop(columns=drop_columns, inplace=True)
        if df.shape[0] <= 600:
            continue

        # Test different feature selection and dimensionality reduction techniques
        for method in reduction_methods:
            if method == 'pearson':
                for k in k_values:
                    rf_results_path = f'data/results/{drug_id}_{method}_{k}_rf.csv'
                    svr_results_path = f'data/results/{drug_id}_{method}_{k}_svr.csv'
                    en_results_path = f'data/results/{drug_id}_{method}_{k}_en.csv'
                    nn_results_path = f'data/results/{drug_id}_{method}_{k}_nn.csv'
                    if os.path.exists(rf_results_path) and os.path.exists(svr_results_path) and os.path.exists(en_results_path) and os.path.exists(nn_results_path):
                        print(f"Result '{drug_id}_{method}_{k}' found. Loading the result...")
                        rf_result = pd.read_csv(rf_results_path)
                        results.append(tuple(rf_result.iloc[0]))
                        svr_result = pd.read_csv(svr_results_path)
                        results.append(tuple(svr_result.iloc[0]))
                        en_result = pd.read_csv(en_results_path)
                        results.append(tuple(en_result.iloc[0]))
                        nn_result = pd.read_csv(nn_results_path)
                        results.append(tuple(nn_result.iloc[0]))
                    else:
                        top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
                        top_features = top_K_pearson_df['Gene'].values
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                            df, top_features, target_variable)
                        
                        evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results, drug_id)
            elif method == 'nmf':
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                for n_components in n_components_list:
                    rf_results_path = f'data/results/{drug_id}_{method}_{k}_rf.csv'
                    svr_results_path = f'data/results/{drug_id}_{method}_{k}_svr.csv'
                    en_results_path = f'data/results/{drug_id}_{method}_{k}_en.csv'
                    nn_results_path = f'data/results/{drug_id}_{method}_{k}_nn.csv'
                    if os.path.exists(rf_results_path) and os.path.exists(svr_results_path) and os.path.exists(en_results_path) and os.path.exists(en_results_path):
                        print(f"Result '{drug_id}_{method}_{k}' found. Loading the result...")
                        rf_result = pd.read_csv(rf_results_path)
                        results.append(tuple(rf_result.iloc[0]))
                        svr_result = pd.read_csv(svr_results_path)
                        results.append(tuple(svr_result.iloc[0]))
                        en_result = pd.read_csv(en_results_path)
                        results.append(tuple(en_result.iloc[0]))
                        nn_result = pd.read_csv(nn_results_path)
                        results.append(tuple(nn_result.iloc[0]))
                    else:
                        W, _, _ = apply_nmf(X, n_components)
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                            X = pd.DataFrame(W), y = y)

                        evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, n_components, method, results, drug_id)
            elif method == 'pca':
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                for n_components in pca_components:
                    rf_results_path = f'data/results/{drug_id}_{method}_{k}_rf.csv'
                    svr_results_path = f'data/results/{drug_id}_{method}_{k}_svr.csv'
                    en_results_path = f'data/results/{drug_id}_{method}_{k}_en.csv'
                    nn_results_path = f'data/results/{drug_id}_{method}_{k}_nn.csv'
                    if os.path.exists(rf_results_path) and os.path.exists(svr_results_path) and os.path.exists(en_results_path) and os.path.exists(en_results_path):
                        print(f"Result '{drug_id}_{method}_{k}' found. Loading the result...")
                        rf_result = pd.read_csv(rf_results_path)
                        results.append(tuple(rf_result.iloc[0]))
                        svr_result = pd.read_csv(svr_results_path)
                        results.append(tuple(svr_result.iloc[0]))
                        en_result = pd.read_csv(en_results_path)
                        results.append(tuple(en_result.iloc[0]))
                        nn_result = pd.read_csv(nn_results_path)
                        results.append(tuple(nn_result.iloc[0]))
                    else:
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X = X, y = y)
                        X_train, X_val, X_test = perform_pca(n_components, X_train, X_val, X_test)

                        evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, n_components, method, results, drug_id)
    print("Evaluation complete.")
    plot_results(results)

def evaluate_models(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results, drug_id):
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
        drug_id (int): drug id data used to evaluate model
    """
    # evaluate Random Forest
    rf_results_path = f'data/results/{drug_id}_{method}_{k}_rf.csv'
    start_time = time.time()
    rf_model = grid_search_random_forest(drug_id, method, k, X_train, y_train)
    rf_mse, _ = evaluate_model(rf_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Random Forest', k, rf_mse, elapsed_time, drug_id))
    df_results = pd.DataFrame(
        [[method, 'Random Forest', k, rf_mse, elapsed_time, drug_id]],
        columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID']
    )
    print(f"Saved result to {rf_results_path}")
    df_results.to_csv(rf_results_path, index=False)

    # evaluate SVR
    svr_results_path = f'data/results/{drug_id}_{method}_{k}_svr.csv'
    start_time = time.time()
    svr_model = grid_search_svr(drug_id, method, k, X_train, y_train)
    svr_mse, _ = evaluate_model(svr_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'SVR', k, svr_mse, elapsed_time, drug_id))
    df_results = pd.DataFrame(
        [[method, 'SVR', k, svr_mse, elapsed_time, drug_id]],
        columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID']
    )
    print(f"Saved result to {svr_results_path}")
    df_results.to_csv(svr_results_path, index=False)
        
    # evaluate Elastic Net
    en_results_path = f'data/results/{drug_id}_{method}_{k}_en.csv'
    start_time = time.time()
    en_model = grid_search_elastic_net(drug_id, method, k, X_train, y_train)
    en_mse, _ = evaluate_model(en_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Elastic Net', k, en_mse, elapsed_time, drug_id))
    df_results = pd.DataFrame(
        [[method, 'Elastic Net', k, en_mse, elapsed_time, drug_id]],
        columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID']
    )
    print(f"Saved result to {en_results_path}")
    df_results.to_csv(en_results_path, index=False)

    # evaluate Neural Network
    nn_results_path = f'data/results/{drug_id}_{method}_{k}_nn.csv'
    start_time = time.time()
    nn_model, _ = train_neural_network(drug_id, method, k, X_train, y_train, X_val, y_val)
    nn_mse, _ = evaluate_model(nn_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Neural Network', k, nn_mse, elapsed_time, drug_id))
    df_results = pd.DataFrame(
        [[method, 'Neural Network', k, nn_mse, elapsed_time, drug_id]],
        columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID']
    )
    print(f"Saved result to {nn_results_path}")
    df_results.to_csv(nn_results_path, index=False)

def plot_results(results):
    """
    Plot the average MSE results across drug IDs for each dimensionality reduction technique and model.

    Args:
        results (list): List of results for each drug ID and reduction method.
    """
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID'])

    # Aggregate MSE by Method, Model, and K/Components
    avg_results = df_results.groupby(['Method', 'Model', 'K/Components']).agg({'MSE': 'mean'}).reset_index()

    # Plot the average MSE for each method and model
    methods = avg_results['Method'].unique()
    for method in methods:
        plt.figure(figsize=(12, 6))
        method_subset = avg_results[avg_results['Method'] == method]
        for model in method_subset['Model'].unique():
            model_subset = method_subset[method_subset['Model'] == model]
            plt.plot(
                model_subset['K/Components'], 
                model_subset['MSE'], 
                marker='o', 
                label=model
            )

        plt.xlabel('Number of Features/Components')
        plt.ylabel('Average Mean Squared Error')
        plt.title(f'Average Model Performance for {method.upper()} Reduction Technique')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./data/plots/avg_mse_{method}.png')
        plt.close()

    print("Graphs saved to './data/plots/' folder.")
    
def plot_results_models(results):
    """
    Plot the average MSE results across drug IDs for each reduction method.
    Creates separate graphs for different models.

    Args:
        results (list): List of results for each drug ID and reduction method.
    """
    # Convert results to a DataFrame
    df_results = pd.DataFrame(results, columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID'])

    # Aggregate MSE by Method, Model, and K/Components
    avg_results = df_results.groupby(['Method', 'Model', 'K/Components']).agg({'MSE': 'mean'}).reset_index()
    
    models = avg_results['Model'].unique()
    for model in models:
        plt.figure(figsize=(12, 6))
        model_subset = avg_results[avg_results['Model'] == model]
        for method in model_subset['Method'].unique():
            method_subset = model_subset[model_subset['Method'] == method]
            plt.plot(
                method_subset['K/Components'], 
                method_subset['MSE'], 
                marker='o', 
                label=method
            )

        plt.xlabel('Number of Features/Components')
        plt.ylabel('Average Mean Squared Error')
        plt.title(f'Average Reduction Method Performance for Random Forest')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'./data/plots/avg_mse_{model}.png')
        plt.close()

    plt.xlabel('Number of Features/Components')
    plt.ylabel('Average Mean Squared Error')
    plt.title(f'Average Reduction Method MSE for Random Forest Model')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'./data/plots/avg_mse_{method}.png')
    plt.close()

    print("Graphs saved to './data/plots/' folder.")

def evaluate_rf(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results, drug_id):
    start_time = time.time()
    rf_model = grid_search_random_forest(drug_id, method, k, X_train, y_train)
    rf_mse, _ = evaluate_model(rf_model, X_test, y_test)
    elapsed_time = time.time() - start_time
    results.append((method, 'Random Forest', k, rf_mse, elapsed_time, drug_id))
    
    results_path = f'{results_directory}{drug_id}_{method}_{k}_rf.csv'
    df_results = pd.DataFrame(
        [[method, 'Random Forest', k, rf_mse, elapsed_time, drug_id]],
        columns=['Method', 'Model', 'K/Components', 'MSE', 'Time', 'Drug ID']
    )
    print(f"Saved result to {results_path}")
    df_results.to_csv(results_path, index=False)

def evaluate_best_rm():
    """
    Evaluates the best reduction method as well as the best amount of components.
    """
    
    # initialize dataset
    dataset = Dataset(
        dataset_name=dataset_name,
        type=type,
        gene_file=gene_file_name,
        IC50_file=drug_file_name,
        data_directory=data_directory,
    )
    
    results = []
    dataset.create_drug_id_list()
    drug_ids = dataset.get_drug_id_list()
    
    drug_ids = drug_ids[0:50]
    for drug_id in drug_ids:
        df = dataset.create_data(drug_id)
        df.drop(columns=drop_columns, inplace=True)
        if df.shape[0] <= 600:
            continue
        for method in reduction_methods:
            if method == 'pearson':
                for k in k_values:
                    pearson_results_path = f'{results_directory}{drug_id}_{method}_{k}_rf.csv'
                    if os.path.exists(pearson_results_path):
                        print(f"Result '{pearson_results_path}' found. Loading the result...")
                        df_result = pd.read_csv(pearson_results_path)
                        results.append(tuple(df_result.iloc[0]))
                    else:
                        top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
                        top_features = top_K_pearson_df['Gene'].values
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                            df, top_features, target_variable)
                        
                        evaluate_rf(X_train, X_val, X_test, y_train, y_val, y_test, k, method, results, drug_id)
            elif method == 'nmf':
                for n_components in n_components_list:
                    nmf_results_path = f'{results_directory}{drug_id}_{method}_{n_components}_rf.csv'
                    if os.path.exists(nmf_results_path):
                        print(f"Result '{nmf_results_path}' found. Loading the result...")
                        df_result = pd.read_csv(nmf_results_path)
                        results.append(tuple(df_result.iloc[0]))
                    else:
                        print(f"Cannot find '{nmf_results_path}'.")
                        X = df.drop(columns=[target_variable])
                        y = df[target_variable]
                        W, _, _ = apply_nmf(X, n_components)
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(
                            X = pd.DataFrame(W), y = y)

                        evaluate_rf(X_train, X_val, X_test, y_train, y_val, y_test, n_components, method, results, drug_id)
            elif method == 'pca':
                X = df.drop(columns=[target_variable])
                y = df[target_variable]
                for n_component in pca_components:
                    pca_results_path = f'{results_directory}{drug_id}_{method}_{n_component}_rf.csv'
                    if os.path.exists(pca_results_path):
                        print(f"Result '{pca_results_path}' found. Loading the result...")
                        df_result = pd.read_csv(pca_results_path)
                        results.append(tuple(df_result.iloc[0]))
                    else:
                        print(f"Cannot find '{pca_results_path}'.")
                        X_train, X_val, X_test, y_train, y_val, y_test = split_data(X = X, y = y)
                        X_train, X_val, X_test = perform_pca(n_component, X_train, X_val, X_test)

                        evaluate_rf(X_train, X_val, X_test, y_train, y_val, y_test, n_component, method, results, drug_id)
        print(f"Evaluated Reduction Methods for {drug_id}")

    print("Found best Reduction Method.")
    plot_results_models(results)

if __name__ == "__main__":
    main()
    # evaluate_best_rm() # run only to find the best feature reduction technique
