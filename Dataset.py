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

