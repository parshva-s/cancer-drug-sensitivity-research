import torch
import torch.nn.functional as F
from sklearn.metrics import mean_squared_error
from torch_geometric.data import Data, DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool


class GNNModel(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(GNNModel, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.fc = torch.nn.Linear(hidden_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = global_mean_pool(x, batch)  # Aggregate node features into graph representation
        x = self.fc(x)
        return x


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

    test_data = create_graph_data(X_test.values, y_test.values)  # Assuming pandas DataFrame/Series
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
