# import pandas as pd
# from scipy.stats import pearsonr

# df = pd.read_csv("data/final_dataset.csv")

# target_variable = 'LN_IC50'
# pearson_correlations = {}

# for gene in df.columns:
#     if gene != target_variable:
#         correlation, _ = pearsonr(df[gene], df[target_variable])
#         pearson_correlations[gene] = abs(correlation)

# pearson_df = pd.DataFrame(list(pearson_correlations.items()), columns=[
#                           'Gene', 'Correlation'])

# pearson_df_sorted = pearson_df.sort_values(by='Correlation', ascending=False)

# top_genes = pearson_df_sorted.head(10)
# print(top_genes)

# selected_genes = pearson_df_sorted['Gene'].head(10)


import pandas as pd
from scipy.stats import pearsonr
pd.set_option('display.max_colwidth', None)
pd.set_option('display.max_rows', None)

df = pd.read_csv("data/final_dataset.csv")

ln_ic50 = 'LN_IC50'
auc = 'AUC'

pearson_ln_ic50 = {}
pearson_auc = {}

for gene in df.columns:
    if gene not in [ln_ic50, auc]:
        if df[gene].ndim > 1:
            df[gene] = df[gene].iloc[:, 0]

        correlation, _ = pearsonr(df[gene], df[ln_ic50])
        pearson_ln_ic50[gene] = abs(correlation)

for gene in df.columns:
    if gene not in [ln_ic50, auc]:
        if df[gene].ndim > 1:
            df[gene] = df[gene].iloc[:, 0]

        correlation, _ = pearsonr(df[gene], df[auc])
        pearson_auc[gene] = abs(correlation)

pearson_ln_ic50_df = pd.DataFrame(
    list(pearson_ln_ic50.items()), columns=['Gene', 'Correlation'])
pearson_ln_ic50_df_sorted = pearson_ln_ic50_df.sort_values(
    by='Correlation', ascending=False)

pearson_auc_df = pd.DataFrame(
    list(pearson_auc.items()), columns=['Gene', 'Correlation'])
pearson_auc_df_sorted = pearson_auc_df.sort_values(
    by='Correlation', ascending=False)

print("Top genes correlated with LN_IC50 (excluding AUC):")
print(pearson_ln_ic50_df_sorted.head(100))

print("Top genes correlated with AUC (excluding LN_IC50):")
print(pearson_auc_df_sorted.head(10))
