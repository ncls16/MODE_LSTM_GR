# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %% Variables

# dossier origine
dir_proj = os.path.normpath(os.getcwd())
# dossier resultat
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
print('dir_results:', dir_results)

# fichier total
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'resultats_totaux.csv'))

# Dossier ou sont stockés les plots
dir_plots = os.path.normpath(os.path.join(dir_proj, f'plots'))
os.makedirs(dir_plots, exist_ok=True)


# %% Fonctions
def csv_df(file: str) -> pd.DataFrame:
    """Convertir un fichier csv en DataFrame"""
    df = pd.read_csv(file)
    df = df.loc[:, df.columns.str.contains('BV|MAE test|NSE test|epoch|seq_len')]
    return df


def plot_moustache(df):
    """Plot la distribution des MAE"""
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.boxplot(df['MAE_test_LSTM'])
    plt.boxplot(df['MAE_test_GR4J'])
    plt.title('Distribution des MAE')
    plt.ylabel('MAE')

    plt.subplot(1, 2, 2)
    plt.boxplot(df['NSE_test_LSTM'])
    plt.boxplot(df['NSE_test_GR4J'])
    plt.title('Distribution des NSE')
    plt.ylabel('NSE')
    plt.savefig(os.path.join(dir_plots, 'boxplot.png'))
    plt.show() # afficher le plot