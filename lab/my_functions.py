import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def remove_outliers(df, col):
    """remove outliers according to Tukeys rule"""
    Q25 = df[col].quantile(.25)
    Q75 = df[col].quantile(.75)
    IQR = Q75-Q25
    lower_bound = Q25 - 1.5 * IQR
    higher_bound = Q75 + 1.5 * IQR
    mask = (df[col] > lower_bound) & (df[col] < higher_bound)
    df = df.loc[mask]
    return df

def scatterplot_boxplot(df, column_name):
    fig, ax = plt.subplots(1, 2, figsize = (8, 4))
    sns.scatterplot(data=df, x=df.index, y= column_name, ax=ax[0]);
    sns.boxplot(data=df, x= column_name, ax=ax[1]);
    return fig