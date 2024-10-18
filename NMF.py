from sklearn.decomposition import NMF
import pandas as pd


def NMF_reduction(dataframe: pd.DataFrame, n_components: int):
    """
    Perform NMF on the input dataframe and return the reduced dataframe
    param dataframes: dataframe to be reduced
    param n_components: the number of components to reduce to
    return: reduced dataframes
    """
    nmf = NMF(n_components=n_components)
    reduced_dataframe = nmf.fit_transform(dataframe)
    return reduced_dataframe

