from preprocessor.Dataset import Dataset
def main():
    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"
    drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']
    target_variable = "LN_IC50"

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

if __name__ == "__main__":
    main()
