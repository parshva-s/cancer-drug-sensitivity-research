import os
import pandas as pd


def load_and_process_files(folder_path, assay_to_cellLine):
    """
    Load CSV files from the specified folder, process gene data, and return the gene means for each file.
    """
    cell_lines_data = {}
    all_genes = set()

    assay_mapping_df = pd.read_csv(assay_to_cellLine)
    assay_to_cellLine_mapping = dict(
        zip(assay_mapping_df['assay_name'], assay_mapping_df['cell_line']))

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            base_filename = filename.removesuffix('.csv')
            df = pd.read_csv(file_path)

            # Filter rows where gene is not null
            df = df[df['gene'].notna()]

            # Calculate mean expression value for each gene
            gene_means = df.groupby('gene')['expression_value'].mean()

            # Determine cell line name based on mapping or fallback to filename
            cell_line_name = assay_to_cellLine_mapping.get(
                base_filename, base_filename)

            # Store results and update unique gene set
            cell_lines_data[cell_line_name] = gene_means
            all_genes.update(gene_means.index)

    return cell_lines_data, all_genes


def create_final_dataframe(cell_lines_data, all_genes):
    """
    Create a DataFrame with genes as columns and cell lines as rows, filling missing values with 0.
    """
    final_df = pd.DataFrame(columns=sorted(all_genes))

    for cell_line, gene_means in cell_lines_data.items():
        row = pd.Series(0, index=final_df.columns)
        row.update(gene_means)
        final_df.loc[cell_line] = row

    return final_df


def main():
    """
    Main function to execute the workflow.
    """
    folder_path = 'data'
    assay_to_cellLine = 'assay_to_cellLine_mapping.csv'
    output_file = 'cell_line_expressions.csv'

    # Load, process files, and create DataFrame
    cell_lines_data, all_genes = load_and_process_files(
        folder_path, assay_to_cellLine)
    final_df = create_final_dataframe(cell_lines_data, all_genes)

    # Save final DataFrame to CSV
    final_df.to_csv(output_file, index_label='Cell_Line')


if __name__ == '__main__':
    main()
