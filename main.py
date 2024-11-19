from preprocessor.Dataset import Dataset
from feature_reduction.NMF_expression import apply_nmf
from utils.common import split_data
from feature_reduction.feature_selection import perform_pearson_correlation


def main():
    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"
    drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']
    target_variable = "LN_IC50"

    # feature parameters
    k_values = [25, 50]
    n_components_list = [15, 20]


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
    for method in ['pearson', 'nmf']:
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
        results (list): list to store results
    """
    

if __name__ == "__main__":
    main()
