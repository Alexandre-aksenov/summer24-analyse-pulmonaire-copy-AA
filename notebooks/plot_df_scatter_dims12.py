import seaborn as sns
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


def plt_scatter_dims12(data: pd.DataFrame, colors: pd.Series):
    """
    Affiche les 2 premiéres colonnes d'une matrice avec la légende.

    Args:
        data (matrix): the data to plot.
        colors (pd.Series): the colors of points.

    Returns:
        Output of sns.scatterplot
    """

    # for checking the data:
    print("The shape of the plotted data:", data.shape)
    print("The shape of the classes:", colors.shape)
    print("The color column is called:", colors.name)


    df_2cols_merge = data.merge(colors,
                                left_index=True,
                                right_index=True,
                                how='inner') if colors.name not in data.columns else data
    
    # for checking the data:
    print("The number of points for each class:")
    print(df_2cols_merge[colors.name].value_counts())

    return sns.scatterplot(data=df_2cols_merge,
                            x=data.columns[0],
                            y=data.columns[1],
                            hue=colors.name,
                            legend="full")


# test, see: 
# https://seaborn.pydata.org/archive/0.11/generated/seaborn.scatterplot.html
if __name__ == '__main__':
    tips = sns.load_dataset("tips")
    sns.scatterplot(data=tips, x="total_bill", y="tip", hue="time")
    plt.show()
    print('----------')
    plt_scatter_dims12(tips.iloc[:, :2], tips["time"])
    plt.show()
    # correct plot!

    print('----------')
    # this example calls the 'else' branch in the function
    plt_scatter_dims12(tips, tips["time"])
    plt.show()
    # correct plot!
    