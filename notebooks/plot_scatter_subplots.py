"""
Scatterplots for each level of target var.

"""


import matplotlib.pyplot as plt

import time
import pandas as pd


# 
def subplots_levels(X_train : pd.DataFrame, y_train: pd.Series):
    """
    A scatterplot with 1st and 2nd columns for each level of the series in 2nd parameter. 

    Args:
        X_train (pd.DataFrame): values
        y_train (pd.Series): levels

    Returns:
        figure, axes.
    """
    all_levels = y_train.value_counts().index

    fig, axes = plt.subplots(len(all_levels), 1)
    fig.set_figheight(16)
    fig.set_figwidth(5)
    
    for ind, lvl in enumerate(all_levels):
        df_level_lvl = X_train.loc[y_train[y_train == lvl].index, :]
        # (Index object for rows for rows with y_train == lvl,
        # see: https://stackoverflow.com/a/52173171 ,
        # convertible TO SUBFUNCTION)
    
        axes[ind].scatter(df_level_lvl.iloc[:, 0], df_level_lvl.iloc[:, 1])
        axes[ind].set_title("Level: " + str(lvl))
    
    return fig, axes


# test, see: 
# https://seaborn.pydata.org/archive/0.11/generated/seaborn.scatterplot.html
if __name__ == '__main__':
    import seaborn as sns
    
    
    tips = sns.load_dataset("tips")
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
    plt.show()
    
    subplots_levels(tips, tips["time"])
    plt.show()
    # Ceci affiche 2 graphiques conforme avec le site seaborn. OK