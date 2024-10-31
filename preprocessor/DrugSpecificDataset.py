from Dataset import Dataset

class DrugSpecificDataset():
    def __init__(self,
        drug_id: str = None,
        features: list = None,
        targets: list = None):
        
        self.drug_id = drug_id

        # dataset parameters
        self.feature_names = None
        self.features = None
        self.targets = None

        self.dataset = None