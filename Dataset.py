import pandas as pd
import numpy as np
import os


class Dataset:
    def __init__(
            self,
            dataset_name,
            gene_file: str = None,
            IC50_file: str = None,
            data_directory: str = None,
            create_data: bool = True):
        self.dataset_name = dataset_name
        self.features = None
        self.targets = None
        self.feature_names = None

        # dataframes used
        self.gene_expression_data = None
        self.drug_cell_line_data = None
        self.dataset = None

        # create dataframes if file names are defined
        if gene_file is not None and data_directory is not None:
            self.set_features_GDSC(gene_file, data_directory)

        if IC50_file is not None and data_directory is not None:
            self.set_targets_GDSC(IC50_file, data_directory)
        
        if create_data:
            self.create_data()
    
    def get_dataset_info(self):
        """Gets the info about the dataset
        """
        if self.dataset_name == "GDSC":
            print("This is the GDSC dataset.")
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

    def set_features_GDSC(self, feature_file_name: str,
                          data_directory: str) -> None:
        """set features from a file

        Args:
            feature_file_name (str): the name of the file to read features from.
            data_directory (str): the directory which feature file is in
        """
        if not self.check_valid_file(feature_file_name, data_directory):
            return

        data = pd.read_csv(data_directory + feature_file_name)

        grouped_data = data.groupby(
            ['COSMIC ID', 'Genes in Segment'], as_index=False).agg({'IS Mutated': 'max'})
        pivoted_data = grouped_data.pivot(
            index='COSMIC ID',
            columns='Genes in Segment',
            values='IS Mutated')

        # Fill any missing values with 0 (if any genes are missing for certain
        # cell lines)
        pivoted_data = pivoted_data.fillna(0)

        # Convert the data to integer type (from float if NaN was present)
        self.gene_expression_data = pivoted_data.astype(int)

        # get list  of data headers
        self.feature_names = self.gene_expression_data.columns

    def set_targets_GDSC(self, target_file: str, data_directory: str) -> None:
        """set targets from a file

        Args:
            feature_file_name (str): the name of the file to read features from.
            data_directory (str): the directory which feature file is in
        """
        if not self.check_valid_file(target_file, data_directory):
            return

    def check_valid_file(file_name: str, data_directory: str) -> bool:
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
        
        acceptable_modes = ["gene", "compound", "final"]
        
        if mode not in acceptable_modes:
            print(f"Invalid mode. Must enter modes from: {acceptable_modes}")
            return
        
        if output_file_name is not None:
            if not output_file_name.endswith(".csv"):
                output_file_name += ".csv"
        else:
            output_file_name = f"{mode}_out.csv"
        


if __name__ == "__main__":
    data_directory = "data/"
    dataset = Dataset("GDSC")
    dataset.set_features_GDSC("gene_expression.csv", data_directory)
    # dataset.gene_data_to_csv(data_directory)
