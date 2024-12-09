import pandas as pd
import numpy as np
import os


class Dataset:
    def __init__(
            self,
            dataset_name: str,
            type: str,
            gene_file: str = None,
            IC50_file: str = None,
            data_directory: str = None):

        self.dataset_name = dataset_name  # name of dataset
        if type not in ["binary", "expression"]:
            print("Invalid type. Must be either 'binary' or 'expression'.")
            return
        self.type = type  # type of dataset

        # dataset parameters
        self.drug_id_list = None
        self.current_drug_id = None
        self.feature_names = None
        self.target_names = None

        # dataframes used
        self.gene_expression_data = None
        self.drug_cell_line_data = None
        self.dataset = None
        self.modes = ["gene", "compound", "final"]

        # create dataframes if file names are defined
        if gene_file is not None and data_directory is not None:
            self.set_features(gene_file, data_directory)

        if IC50_file is not None and data_directory is not None:
            self.set_targets(IC50_file, data_directory)
        
        if self.drug_cell_line_data is not None:
            self.create_drug_id_list()

        # if create_data:
        #     self.create_data()

    def get_dataset_info(self):
        """Gets the info about the dataset
        """
        if self.dataset_name == "GDSC2":
            print("This is the GDSC2 dataset.")
        elif self.dataset_name == "ArrayExpress":
            print("This is the ArrayExpress dataset.")
        elif self.dataset_name == "CCLE":
            print("This is the CCLE dataset.")
        else:
            print("No dataset was defined.")

    def get_feature_names(self) -> list:
        """get names of features

        Returns:
            list: feature names
        """
        return self.feature_names

    def display_feature_names(self):
        """print name of features
        """
        for name in self.feature_names:
            print(name)

    def get_feature_size(self) -> dict:
        """gets the size of the feature space

        Returns:
            dict: the x and y dimensions of feature space
        """
        return self.feature_size

    def set_features(self, feature_file_name: str,
                          data_directory: str) -> None:
        """set features from a file

        Args:
            feature_file_name (str): the name of the file to read features from.
            data_directory (str): the directory which feature file is in
        """
        if not self.check_valid_file(feature_file_name, data_directory):
            return

        data = pd.read_csv(data_directory + feature_file_name)
        
        # set binary matrix data
        if self.type == "binary":
            grouped_data = data.groupby(
                ['Cell Line Name', 'Genes in Segment'], as_index=False).agg({'IS Mutated': 'max'})
            pivoted_data = grouped_data.pivot(
                index='Cell Line Name',
                columns='Genes in Segment',
                values='IS Mutated')

            # Rename the first column "Cell Line Name" to "sample"
            self.gene_expression_data.rename(columns={"Cell Line Name": "Cell_Line"}, inplace=True)
            pivoted_data.set_index("Cell_Line", inplace=True)
            pivoted_data = pivoted_data.fillna(0)

            # Convert the data to integer type (from float if NaN was present)
            self.gene_expression_data = pivoted_data.astype(int)
        
        # set gene expression data
        if self.type == "expression":
            # check if column with cell line names exists
            if self.dataset_name == "CCLE":
                # rename the first column to "Cell_Line"
                data.rename(columns={data.columns[0]: "Cell_Line"}, inplace=True)
                # map cell line to id in first column to cell line name in cell_line_mapping.csv which contains the id and drug names
                cell_line_mapping = pd.read_csv(data_directory + "cell_line_mapping_ccle.csv")
                cell_line_mapping = dict(zip(cell_line_mapping["ModelID"], cell_line_mapping["CellLineName"]))
                data["Cell_Line"] = data["Cell_Line"].map(cell_line_mapping)
                
            self.gene_expression_data = data.set_index("Cell_Line")

        # Rename gene columns to generic format (e.g., "gene_1", "gene_2", ...)
        self.gene_expression_data.columns = [f"gene_{i+1}" for i in range(data.shape[1]-1)]
        
        # get list  of data headers
        self.feature_names = self.gene_expression_data.columns.tolist()

    def set_targets(self, target_file: str, data_directory: str) -> None:
        """set targets from a file

        Args:
            feature_file_name (str): the name of the file to read targets from.
            data_directory (str): the directory which target file is in
        """
        if not self.check_valid_file(target_file, data_directory):
            return

        data = pd.read_csv(data_directory + target_file)

        self.target_names = [
            "CELL_LINE_NAME",
            "DRUG_ID",
            "LN_IC50",
            "AUC",
            "RMSE",
            "Z_SCORE"]

        data = data[self.target_names]
        self.drug_cell_line_data = data

    def create_data(self, drug_id):
        """Create a dataset for a specific drug ID without using chunks."""
        if self.gene_expression_data.empty or self.drug_cell_line_data.empty:
            print("Data has not been defined yet. Cannot create final dataset.")
            return

        print(f"Creating dataset for drug ID: {drug_id}...")

        # Filter the drug-cell line data for the specific drug ID
        filtered_data = self.drug_cell_line_data[self.drug_cell_line_data['DRUG_ID'] == drug_id]
        if filtered_data.empty:
            print(f"No data found for drug ID: {drug_id}")
            self.dataset = pd.DataFrame()
            return
        
        self.current_drug_id = drug_id
        
        # Merge the filtered data with gene expression data
        merged_dataset = pd.merge(
            self.gene_expression_data,
            filtered_data,
            left_on="Cell_Line",
            right_on="CELL_LINE_NAME",
            how="left"
        )
        
        self.dataset = merged_dataset.dropna(subset=["LN_IC50"])
        
        return self.dataset
            
    def create_data_from_csv(self, directory : str, file_name : str):
        """create the dataset from a csv file

        Args:
            directory (str): directory of the file
            file_name (str): name of the file
        """
        if not self.check_valid_file(file_name, directory):
            return

        self.dataset = pd.read_csv(directory + file_name)
        self.create_drug_id_dict()
    
    def create_drug_id_list(self):
        """create a list of drug ids"""
        print("Creating drug id list...")
        
        self.drug_id_list = self.drug_cell_line_data['DRUG_ID'].unique().tolist()
    
    def check_valid_file(self, file_name: str, data_directory: str) -> bool:
        """checks if a file is valid

        Args:
            data_directory (str): the directory of the file
            file_name (str): the name of the file

        Returns:
            bool: True if file is valid, False otherwise
        """
        if file_name is None:
            print("No file name provided")
            return False

        if not os.path.exists(data_directory + file_name):
            print("File does not exist")
            return False

        if not file_name.endswith(".csv"):
            print("File is not a csv file")
            return False

        return True

    def data_to_csv(
            self,
            mode: str,
            data_directory: str,
            output_file_name: str = None) -> None:
        """creates a csv file from gene expression data

        Args:
            mode (str): different modes to save different data files
            data_directory (str): the directory of where csv will be stored
            output_file_name (str, optional): name of the file to be stored to. Defaults to None.
        """
        if mode not in self.modes:
            print(f"Invalid mode. Must enter modes from: {self.modes}")
            return

        if output_file_name is not None:
            if not output_file_name.endswith(".csv"):
                output_file_name += ".csv"
        else:
            output_file_name = f"{mode}_out.csv"

        match mode:
            case "gene":
                if self.gene_expression_data.empty:
                    print("Gene expression data not set.")
                    return
                self.gene_expression_data.to_csv(
                    data_directory + output_file_name)
            case "compound":
                if self.drug_cell_line_data.empty:
                    print("Drug cell line data not set.")
                    return
                self.drug_cell_line_data.to_csv(
                    data_directory + output_file_name)
            case "final":
                if self.dataset.empty:
                    print("Final data not set.")
                    return
                # test, only print the first drug id data
                self.dataset.to_csv(f"{data_directory}{self.current_drug_id}_{output_file_name}")
        print(f"{mode} data saved in: {data_directory}{self.current_drug_id}_{output_file_name}")

    def get_modes(self) -> list:
        """returns available modes

        Returns:
            list: available modes
        """
        return self.modes

    def get_drug_id_list(self) -> list:
        """returns the drug ids

        Returns:
            list: drug ids
        """
        return self.drug_id_list


def create_csv(
        modes: list,
        dataset: Dataset,
        csv_names: list,
        csv_directory: str = "data/") -> None:
    """creates csv files for defined

    Args:
        modes (list): list of modes for csv files to create
        dataset (Dataset): dataset object that has all the datasets
        csv_names (list): names of csv files to save each mode to
        csv_directory (str, optional): Directory to save csv files to. Defaults to "data/".
    """
    for i in range(len(modes)):
        if i < len(csv_names):
            dataset.data_to_csv(modes[i], csv_directory, csv_names[i])
        else:
            dataset.data_to_csv(modes[i], csv_directory)


def main():
    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"

    # create dataset to be used
    dataset = Dataset(
        dataset_name,
        type,
        gene_file_name,
        drug_file_name,
        data_directory)
    
    print(dataset.create_data(drug_id))
    
    # create_csv(["final"], dataset, [f"final_drug_{drug_id}_dataset.csv"],
    #            data_directory)  # create csv for final dataset

if __name__ == "__main__":
    main()
