from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import os
import sys


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(CURRENT_DIR))


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = torch.nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        x = self.fc(x)
        return x.squeeze()


def tm(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, data.y)
    loss.backward()
    optimizer.step()
    return loss.item()


def em(model, data):
    model.eval()
    with torch.no_grad():
        output = model(data)
        mse = mean_squared_error(data.y.numpy(), output.numpy())
        auc = roc_auc_score(data.y.numpy(), output.numpy())
    return mse, auc


if __name__ == "__main__":
    from preprocessor.Dataset import Dataset
    from feature_reduction.feature_selection import perform_pearson_correlation
    from utils.common import split_data

    dataset_name = "CCLE"
    drug_id = 1003
    type = "expression"
    data_directory = "data/"
    gene_file_name = "cell_line_expressions.csv"
    drug_file_name = "drug_cell_line.csv"
    drop_columns = ['AUC', 'Z_SCORE', 'RMSE', 'CELL_LINE_NAME', 'DRUG_ID']

    dataset = Dataset(
        dataset_name,
        type,
        gene_file_name,
        drug_file_name,
        data_directory)

    k_values = [100]
    target_variable = "LN_IC50"

    df = dataset.create_data(drug_id)
    df.drop(columns=drop_columns, inplace=True)

    for k in k_values:
        top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
        top_features = top_K_pearson_df['Gene'].values

        X_train, X_test, y_train, y_test = split_data(
            df, top_features, target_variable)

        num_nodes = X_train.shape[0]
        edge_index = torch.combinations(torch.arange(
            num_nodes), r=2).T  # Fully connected graph

        # Convert data to PyTorch tensors
        x = torch.tensor(X_train, dtype=torch.float)
        y = torch.tensor(y_train, dtype=torch.float)
        edge_index = torch.tensor(edge_index, dtype=torch.long)

        # Create PyTorch Geometric data object
        data = Data(x=x, edge_index=edge_index)

        # Initialize model, optimizer, and loss function
        input_dim = X_train.shape[1]
        model = GNNModel(input_dim)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.MSELoss()

        # Hyperparameter tuning
        best_mse = float('inf')
        best_auc = 0
        best_params = {}

        for lr in [0.01, 0.001, 0.0001]:
            for hidden_dim in [32, 64, 128]:
                model = GNNModel(input_dim)
                optimizer = torch.optim.Adam(model.parameters(), lr=lr)

                # Training loop
                # You can increase epochs for better results
                for epoch in tqdm(range(100)):
                    loss = tm(model, data, optimizer, criterion)

                # Evaluation
                mse, auc = em(model, data)
                print(
                    f"LR: {lr}, Hidden Dim: {hidden_dim}, MSE: {mse}, AUC: {auc}")

                if mse < best_mse and auc > best_auc:
                    best_mse = mse
                    best_auc = auc
                    best_params = {'lr': lr, 'hidden_dim': hidden_dim}

        print("Best Parameters:", best_params)
        print("Best MSE:", best_mse)
        print("Best AUC:", best_auc)
