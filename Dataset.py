class Dataset:
    def __init__(self, gene_file, IC50_file = None):
        self.features = None
        self.targets = None
        self.feature_names = None
        
        self.feature_size = {"x" : None, "y" : None}
