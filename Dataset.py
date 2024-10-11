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
        
        self.dataset_name = dataset_name # name of dataset
        
        # dataset parameters
        self.features = None
        self.targets = None
        self.feature_names = None

        # dataframes used
        self.gene_expression_data = None
        self.drug_cell_line_data = None
        self.dataset = None
        self.modes = ["gene", "compound", "final"]

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
        if self.dataset_name == "GDSC2":
            print("This is the GDSC2 dataset.")
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
            feature_file_name (str): the name of the file to read targets from.
            data_directory (str): the directory which target file is in
        """
        if not self.check_valid_file(target_file, data_directory):
            return

        data = pd.read_csv(data_directory + target_file)

        feature_names = [
            "COSMIC_ID",
            "DRUG_ID",
            "LN_IC50",
            "AUC",
            "RMSE",
            "Z_SCORE"]
        
        data = data[feature_names]
        self.drug_cell_line_data = data

    def create_data(self):
        """create the dataset from the features and targets
        """
        if self.gene_expression_data.empty or self.drug_cell_line_data.empty:
            print("Data has not been defined yet. Cannot create final dataset.")

        # TODO: fix COSMIC ID column header not having name changed and join on "COSMIC_ID"
        self.dataset = pd.merge(self.gene_expression_data, self.drug_cell_line_data, left_on="COSMIC ID", right_on="COSMIC_ID", how="left")

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
                self.gene_expression_data.to_csv(data_directory + output_file_name)
            case "compound":
                if self.drug_cell_line_data.empty:
                    print("Drug cell line data not set.")
                    return
                self.drug_cell_line_data.to_csv(data_directory + output_file_name)
            case "final":
                if self.dataset.empty:
                    print("Final data not set.")
                    return
                self.dataset.to_csv(data_directory + output_file_name)
        print(f"{mode} data saved in: {data_directory}{output_file_name}")
    
    def get_modes(self) -> list:
        """returns available modes

        Returns:
            list: available modes
        """
        return self.modes


def create_csv(modes : list, dataset : Dataset, csv_names : list, csv_directory : str = "data/") ->  None:
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
    dataset_name = "GDSC2"
    data_directory = "data/"
    gene_file_name = "gene_expression.csv"
    drug_file_name = "drug_cell_line.csv"
    create_final_dataset = True
    
    # create dataset to be used
    dataset = Dataset(dataset_name, gene_file_name, drug_file_name, data_directory, create_final_dataset)
    create_csv(["final"], dataset, ["final_dataset.csv"], data_directory) # create csv for final dataset

if __name__ == "__main__":
    main()
