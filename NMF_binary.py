import numpy as np
import pandas as pd
from Dataset import Dataset
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error, silhouette_score, accuracy_score
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def apply_nmf(X, n_components=20):
    """Apply NMF and return the reduced data and components."""
    nmf = NMF(n_components=n_components, init='random', random_state=42)
    W = nmf.fit_transform(X)
    H = nmf.components_
    return W, H, nmf

def calculate_reconstruction_error(X, W, H):
    """Calculate reconstruction error after NMF."""
    X_reconstructed = W @ H
    error = mean_squared_error(X, X_reconstructed)
    return error

def perform_clustering(W, n_clusters=5):
    """Cluster the reduced data and calculate the silhouette score."""
    kmeans = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
    clusters = kmeans.fit_predict(W)
    silhouette = silhouette_score(W, clusters)
    return clusters, silhouette

def predict_drug_response(W, y):
    """Predict drug response using the reduced features."""
    X_train, X_test, y_train, y_test = train_test_split(W, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = mean_squared_error(y_test, y_pred)
    return accuracy

def nmf_pipeline(dataset, n_components_list):
    X = dataset.features  # The binary mutation data
    y = dataset.targets  # Drug response (can choose AUC, LN_IC50, etc.)

    # Lists to store results for each n_components value
    reconstruction_errors = []
    accuracies_nmf = []
    accuracies_pca = []
    
    # Loop through the n_components values and evaluate NMF
    for n_components in n_components_list:
        print(f"Applying NMF with {n_components} components...")
        
        W, H, nmf = apply_nmf(X, n_components)

        # Calculate reconstruction error
        reconstruction_error = calculate_reconstruction_error(X, W, H)
        print(f"Reconstruction Error (NMF) for {n_components} components: {reconstruction_error}")
        reconstruction_errors.append(reconstruction_error)
        
        # Predict drug response using reduced features (NMF)
        accuracy_nmf = predict_drug_response(W, y)
        print(f"MSE (NMF) for {n_components} components: {accuracy_nmf}")
        accuracies_nmf.append(accuracy_nmf)
        
     # Plot the results
    plt.figure(figsize=(12, 8))

    # Plot reconstruction error vs n_components
    plt.subplot(2, 2, 1)
    plt.plot(n_components_list, reconstruction_errors, marker='o')
    plt.title('Reconstruction Error vs N Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Reconstruction Error')

    # Plot accuracy of drug response prediction (NMF) vs n_components
    plt.subplot(2, 2, 2)
    plt.plot(n_components_list, accuracies_nmf, marker='o', color='green')
    plt.title('Prediction Accuracy (NMF) vs N Components')
    plt.xlabel('Number of Components')
    plt.ylabel('Mean Squared Error')
    
    plt.tight_layout()
    plt.show()

def main():
    # Initialize dataset
    dataset_name = "GDSC2"
    data_directory = "data/"
    gene_file_name = "gene_expression.csv"
    drug_file_name = "drug_cell_line.csv"

    # Create the dataset object
    dataset = Dataset(
        dataset_name=dataset_name,
        gene_file=gene_file_name,
        IC50_file=drug_file_name,
        data_directory=data_directory,
        create_data=True
    )
    
    # Define a list of n_components values to test
    n_components_list = [20, 40, 60, 80, 100, 120, 140]
    
    # Apply the NMF pipeline to the dataset
    nmf_pipeline(dataset, n_components_list)

if __name__ == "__main__":
    main()
