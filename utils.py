import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def get_best_score(grid):
    best_score = -grid.best_score_
    
    print(f'Best score: {best_score:.5f}')
    print(f'Best params: {grid.best_params_}')
    print(f'Best estimator: {grid.best_estimator_}')

    return best_score

def print_cols_large_corr(df, number_cols, target):
    corr = df.corr()
    corr_abs = corr.abs()

    print(corr_abs.nlargest(number_cols, target)[target])

def plot_corr_matrix(df, number_cols, target):
    corr = df.corr(numeric_only=True)

    cols = (
        corr[target]
        .abs()
        .nlargest(number_cols)
        .index
    )

    corr_matrix = corr.loc[cols, cols]

    plt.figure(figsize=(number_cols / 1.5, number_cols / 1.5))

    sns.heatmap(
        corr_matrix,
        linewidths=1.5,
        annot=True,
        square=True,
        fmt='.2f',
        annot_kws={'size': 10}
    )

    plt.show()
