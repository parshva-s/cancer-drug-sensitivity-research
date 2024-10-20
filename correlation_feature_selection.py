from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
from scipy.stats import pearsonr


def load_and_prepare_data(file_path: str, drop_columns: list, target_variable: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and drop specified columns.
    """
    df = pd.read_csv(file_path)
    df.drop(columns=drop_columns, inplace=True)
    return df


def perform_pearson_correlation(df: pd.DataFrame, target_variable: str, k: int = 200) -> pd.DataFrame:
    """
    Perform Pearson correlation for all features in the dataframe and return the top K features.
    """
    pearson_correlations = {}
    for gene in df.columns:
        if gene != target_variable:
            correlation, _ = pearsonr(df[gene], df[target_variable])
            pearson_correlations[gene] = abs(correlation)

    pearson_df = pd.DataFrame(list(pearson_correlations.items()), columns=[
                              'Gene', 'Correlation'])
    pearson_df_sorted = pearson_df.sort_values(
        by='Correlation', ascending=False)
    return pearson_df_sorted.head(k)


def split_data(df: pd.DataFrame, top_features: list, target_variable: str, test_size: float = 0.2):
    """
    Split the data into training and test sets based on the top features and target variable.
    """
    X = df[top_features]
    y = df[target_variable]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    return X_train, X_test, y_train, y_test


def train_random_forest(X_train, y_train, n_estimators: int = 100):
    """
    Train a Random Forest regressor with the given training data.
    """
    rf_regressor = RandomForestRegressor(
        n_estimators=n_estimators, random_state=42)
    rf_regressor.fit(X_train, y_train)
    return rf_regressor


def evaluate_model(rf_regressor, X_test, y_test):
    """
    Evaluate the Random Forest model and return the Mean Squared Error and R² score.
    """
    y_pred = rf_regressor.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2


def print_feature_importances(rf_regressor, top_features: list):
    """
    Print the feature importances from the Random Forest model.
    """
    importances = rf_regressor.feature_importances_
    for i, feature in enumerate(top_features):
        print(f"Feature: {feature}, Importance: {importances[i]:.4f}")


file_path = "data/final_dataset.csv"
target_variable = 'LN_IC50'
k = 10
test_size = 0.2
n_estimators = 100

df = load_and_prepare_data(
    file_path, ['AUC', 'Z_SCORE', 'RMSE'], target_variable)

top_K_pearson_df = perform_pearson_correlation(df, target_variable, k)
top_features = top_K_pearson_df['Gene'].values

X_train, X_test, y_train, y_test = split_data(
    df, top_features, target_variable, test_size)

rf_regressor = train_random_forest(X_train, y_train, n_estimators)

mse, r2 = evaluate_model(rf_regressor, X_test, y_test)

print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# print_feature_importances(rf_regressor, top_features)


# df = pd.read_csv("data/final_dataset.csv")

# df.drop(columns=['AUC', 'Z_SCORE', 'RMSE'], inplace=True)
# target_variable = 'LN_IC50'
# top_K_pearson_df = perform_pearson_correlation(df, 'LN_IC50', 10)
# print(top_K_pearson_df.columns)

# top_features = top_K_pearson_df['Gene'].values

# X = df[top_features]
# y = df[target_variable]

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, test_size=0.2, random_state=42)

# rf_regressor = RandomForestRegressor(n_estimators=100, random_state=42)
# rf_regressor.fit(X_train, y_train)
# y_pred = rf_regressor.predict(X_test)

# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print(f"Mean Squared Error (MSE): {mse:.4f}")
# print(f"R² Score: {r2:.4f}")

# importances = rf_regressor.feature_importances_
# for i, feature in enumerate(top_features):
#     print(f"Feature: {feature}, Importance: {importances[i]:.4f}")
