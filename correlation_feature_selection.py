import pandas as pd
from scipy.stats import pearsonr


def perform_pearson_correlation(df: pd.DataFrame, target_variable: str, k: int = 200) -> pd.DataFrame:
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


df = pd.read_csv("data/final_dataset.csv")

df.drop(columns=['AUC', 'Z_SCORE', 'RMSE'], inplace=True)
target_variable = 'LN_IC50'
top_K_pearson_df = perform_pearson_correlation(df, 'LN_IC50', 10)
print(top_K_pearson_df.columns)
