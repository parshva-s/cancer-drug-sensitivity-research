import os
import pandas as pd


def load_and_process_files(folder_path):
    """
    Load CSV files from the specified folder, process gene data, and return the gene means for each file.
    """
    cell_lines_data = {}
    all_genes = set()

    for filename in os.listdir(folder_path):
        if filename.endswith('.csv'):
            file_path = os.path.join(folder_path, filename)
            df = pd.read_csv(file_path)

            # Filter rows where gene is not null
            df = df[df['gene'].notna()]

            # Calculate mean expression value for each gene
            gene_means = df.groupby('gene')['expression_value'].mean()

            # Store results and update unique gene set
            cell_lines_data[filename] = gene_means
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
    output_file = 'cell_line_expressions.csv'

    # Load, process files, and create DataFrame
    cell_lines_data, all_genes = load_and_process_files(folder_path)
    final_df = create_final_dataframe(cell_lines_data, all_genes)

    # Save final DataFrame to CSV
    final_df.to_csv(output_file, index_label='Cell_Line')


if __name__ == '__main__':
    main()
