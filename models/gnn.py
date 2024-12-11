import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, DataLoader, Batch
from torch_geometric.nn import GCNConv, global_mean_pool, Sequential
import torch.nn as nn


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        torch.manual_seed(42)
        self.dropout_p = 0.1

        seq_arch = [
            (GCNConv(in_channels=1, out_channels=128), 'x, edge_index -> x1'),
            nn.ReLU(inplace=True),
            (global_mean_pool, 'x1, batch -> x2'),
            nn.Linear(128, 64),
            nn.ReLU(),
        ]
        # Cell-line graph branch. Obtains node embeddings.
        self.cell_emb = Sequential('x, edge_index, batch', seq_arch)

        self.fcn = nn.Sequential(
            nn.Linear(65, 32),
            nn.ReLU(),
            nn.Dropout(self.dropout_p),
            nn.Linear(32, 1),
        )

    def forward(self, data):
        X, edge_index, batch_num = data.x.float(), data.edge_index, data.batch
        drug_data = data.y.view(-1, 1).float()
        cell_emb = self.cell_emb(X, edge_index, batch_num)

        assert cell_emb.size(0) == drug_data.size(0), "Mismatch in batch size between cell_emb and drug_emb"

        # Concatenate along the feature dimension
        concat = torch.cat([cell_emb, drug_data], dim=-1)

        # Fully connected network
        y = self.fcn(concat).reshape(-1)
        return y


def create_graph_data(X, y):
    """
       Convert feature matrix and labels into a list of PyTorch Geometric Data objects.

       Args:
           X (np.ndarray or pd.DataFrame): Feature matrix with shape (n_samples, n_features).
           y (np.ndarray or pd.Series): Labels with shape (n_samples,).

       Returns:
           List[Data]: List of PyTorch Geometric Data objects.
   """
    num_genes = X.shape[1]

    # Create a fully connected graph for simplicity
    edge_index = torch.tensor(
        [[i, j] for i in range(num_genes) for j in range(num_genes)],
        dtype=torch.long
    ).t().contiguous()  # Shape [2, num_edges]

    data_list = []
    for i in range(X.shape[0]):
        node_features = torch.tensor(X[i], dtype=torch.float).unsqueeze(-1)  # Shape [num_genes, 1]
        label = torch.tensor([y[i]], dtype=torch.float)  # Graph label
        data = Data(x=node_features, edge_index=edge_index, y=label)
        data_list.append(data)

    return data_list


def evaluate_gnn_model(model, X_test, y_test, batch_size=32):
    """
    Evaluate the GNN model on the test dataset and compute MSE.

    Args:
        model (torch.nn.Module): Trained GNN model.
        X_test (np.ndarray or pd.DataFrame): Feature matrix for test data.
        y_test (np.ndarray or pd.Series): Ground truth labels for test data.
        batch_size (int, optional): Batch size for DataLoader. Defaults to 32.

    Returns:
        float: Mean Squared Error (MSE) on the test dataset.
    """
    # Ensure the model is in evaluation mode
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    test_data = create_graph_data(X_test, y_test.values)  # Assuming pandas DataFrame/Series
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize predictions and ground truth lists
    predictions = []
    ground_truth = []

    # Evaluate on test data
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            output = model(data).squeeze()  # Predictions
            predictions.extend(output.cpu().numpy())
            ground_truth.extend(data.y.cpu().numpy())

    # Calculate Mean Squared Error
    mse = mean_squared_error(ground_truth, predictions)
    return mse
