class Dataset:
    def __init__(self, gene_file, IC50_file=None):
        self.features = None
        self.targets = None
        self.feature_names = None

        self.feature_size = {"x": None, "y": None}

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
    
