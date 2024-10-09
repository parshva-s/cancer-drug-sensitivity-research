import pandas as pd
import numpy as np
import os

class Dataset:
    def __init__(self, gene_file : str = None, IC50_file : str = None, data_directory : str =None):
        self.features = None
        self.targets = None
        self.feature_names = None
        
        self.gene_expression_data = None

        self.feature_size = {"x": None, "y": None}
        
        if gene_file is not None and data_directory is not None:
            self.set_features(gene_file, data_directory)

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
    
    def set_features_GDSC(self, feature_file_name : str, data_directory : str) -> None:
        """set features from a file

        Args:
            feature_file_name (str, optional): the name of the file to read features from. Defaults to None.
        """
        if feature_file_name is None:
            print("No file name provided")
            return
        
        # check that path to file exists
        if not os.path.exists(data_directory + feature_file_name):
            print("File does not exist")
            return
        
        # check file is a csv file
        if not feature_file_name.endswith(".csv"):
            print("File is not a csv file")
            return
        
        data = pd.read_csv(data_directory + feature_file_name)
        
        grouped_data = data.groupby(['Cell Line Name', 'Genes in Segment'], as_index=False).agg({'IS Mutated': 'max'})
        pivoted_data = grouped_data.pivot(index='Cell Line Name', columns='Genes in Segment', values='IS Mutated')

        # Fill any missing values with 0 (if any genes are missing for certain cell lines)
        pivoted_data = pivoted_data.fillna(0)

        # Convert the data to integer type (from float if NaN was present)
        self.gene_expression_data = pivoted_data.astype(int)
            
