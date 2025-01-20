# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %% Variables
fig_size = (10, 5) # taille des figures
binss = 100 # nombre de bins pour les histogrammes
# dossier origine
dir_proj = os.path.normpath(os.getcwd())
# dossier resultat
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
print('dir_results:', dir_results)

# fichier total
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'resultats_totaux.csv'))
if not os.path.exists(fichier_resultat):
    print(f'Le fichier {fichier_resultat} n\'existe pas')

# Dossier ou sont stockés les plots
dir_plots = os.path.normpath(os.path.join(dir_proj, f'plots'))
os.makedirs(dir_plots, exist_ok=True)


# %% Fonctions
def csv_df(file: str) -> pd.DataFrame:
    """Convertir un fichier csv en DataFrame"""
    df = pd.read_csv(file)
    # list_columns_to_expect =
    # df = df.loc[:, df.columns.str.contains('BV|MAE test|NSE test|epoch|seq_len')]

    return df

# %%% seq_len
def plot_moustaches(df):
    """Plot la distribution des MAE et NSE"""
    seq_len = df['seq_len'].unique()

    timed = plt.figure(figsize=fig_size)
    timed.title('Distribution des temps de calcul')
    timed.ylabel('Temps de calcul (s)')
    timed.boxplot(df.loc[df['best_MAE'] == True, 'training_time_GR4J'], labels=['GR4J'])
    timed.boxplot(df.loc[df['best_MAE'] == True, 'training_time_LSTM'], labels=['LSTM_best_MAE'])
    timed.boxplot(df.loc[df['best_NSE'] == True, 'training_time_LSTM'], labels=['LSTM_best_NSE'])
    timed.yscale('log')

    mae = plt.figure(figsize=fig_size)
    mae.title(f'Distribution des MAE pour différents seq_len')
    mae.ylabel('MAE')
    mae.boxplot(df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], labels=['GR4J'])
    mae.boxplot(df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], labels=['LSTM_best_MAE'])

    nse = plt.figure(figsize=fig_size)
    nse.title(f'Distribution des NSE pour différents seq_len')
    nse.ylabel('NSE')
    nse.boxplot(df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], labels=['GR4J'], showfliers=False)
    nse.boxplot(df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], labels=['LSTM_best_NSE'], showfliers=False)

    for sl in seq_len:
        timed.boxplot(df.loc[df['seq_len'] == sl, 'training_time'], labels=[f'LSTM_{sl}'])
        mae.boxplot(df.loc[df['seq_len'] == sl, 'MAE_test_LSTM'], labels=[f'LSTM_MAE_{sl}'])
        nse.boxplot(df.loc[df['seq_len'] == sl, 'NSE_test_LSTM'], labels=[f'LSTM_NSE_{sl}'])

    timed.savefig(os.path.join(dir_plots, 'boxplot_time.png'))
    mae.savefig(os.path.join(dir_plots, 'boxplot_MAE.png'))
    nse.savefig(os.path.join(dir_plots, 'boxplot_NSE.png'))


# def plot_histogram(df):
#     """Plot l'histogramme des MAE et NSE"""
#     """les best et ou seq_len a ajouter"""
#     mean_time = df.groupby('seq_len')['training_time_LSTM'].mean()
#     mean_mae = df.groupby('seq_len')['MAE_test_LSTM'].mean()
#     mean_nse = df.groupby('seq_len')['NSE_test_LSTM'].mean()
#     mean_time.append(df['training_time_GR4J'].mean())
#     mean_mae.append(df['MAE_test_GR4J'].mean())
#     mean_nse.append(df['NSE_test_GR4J'].mean())

#     hist_time = plt.figure(figsize=(10, 5))
#     hist_time.title('Temps de calcul moyen')
#     hist_time.ylabel('Temps de calcul (s)')

#     return(True)

# %%% Surfaces
def plot_surfaces(df):
    """Plot la répartition des surfaces et la MAE/NSE en fonction de la surface"""
    repartition = plt.figure(figsize=fig_size)
    repartition.title('Répartition des surfaces')
    repartition.ylabel('Nombre de BV')
    repartition.xlabel('Surface (km²)')
    repartition.hist(df.loc[df['seq_len'] == 7, 'AREA_SQKM_'], bins=binss)
    repartition.savefig(os.path.join(dir_plots, 'surfaces_population.png'))

    mae = plt.figure(figsize=fig_size)
    mae.title('MAE en fonction de la surface')
    mae.ylabel('MAE')
    mae.xlabel('Surface (km²)')
    mae.scatter(df.loc[df['best_MAE'] == True, 'AREA_SQKM_'], df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    mae.scatter(df.loc[df['best_MAE'] == True, 'AREA_SQKM_'], df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    mae.legend()
    mae.grid(True)
    mae.grid(True, which='minor')
    mae.xscale('log')
    mae.savefig(os.path.join(dir_plots, 'surfaces_MAE.png'))

    nse = plt.figure(figsize=fig_size)
    nse.title('NSE en fonction de la surface')
    nse.ylabel('NSE')
    nse.xlabel('Surface (km²)')
    nse.scatter(df.loc[df['best_NSE'] == True, 'AREA_SQKM_'], df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    nse.scatter(df.loc[df['best_NSE'] == True, 'AREA_SQKM_'], df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    nse.legend()
    nse.xscale('log')
    nse.grid(True)
    nse.grid(True, which='minor')
    nse.savefig(os.path.join(dir_plots, 'surfaces_NSE.png'))

# %%% Aridite potentielle
def plot_aridite(df):
    """Plot la répartition de l'aridité et la MAE/NSE en fonction de l'aridité"""
    repartition = plt.figure(figsize=fig_size)
    repartition.title('Répartition de l\'aridité')
    repartition.ylabel('Nombre de BV')
    repartition.xlabel('Aridité')
    repartition.hist(df.loc[df['seq_len'] == 7, 'Arid'], bins=binss)
    repartition.savefig(os.path.join(dir_plots, 'aridite_population.png'))

    mae = plt.figure(figsize=fig_size)
    mae.title('MAE en fonction de l\'aridité')
    mae.ylabel('MAE')
    mae.xlabel('Aridité')
    mae.scatter(df.loc[df['best_MAE'] == True, 'Arid'], df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    mae.scatter(df.loc[df['best_MAE'] == True, 'Arid'], df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    mae.legend()
    mae.grid(True)
    mae.savefig(os.path.join(dir_plots, 'aridite_MAE.png'))

    nse = plt.figure(figsize=fig_size)
    nse.title('NSE en fonction de l\'aridité')
    nse.ylabel('NSE')
    nse.xlabel('Aridité')
    nse.scatter(df.loc[df['best_NSE'] == True, 'Arid'], df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    nse.scatter(df.loc[df['best_NSE'] == True, 'Arid'], df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    nse.legend()
    nse.grid(True)
    nse.savefig(os.path.join(dir_plots, 'aridite_NSE.png'))

# %%% Neige
def plot_neige(df):
    """Plot la répartition de la neige et la MAE/NSE en fonction de la neige"""
    repartition = plt.figure(figsize=fig_size)
    repartition.title('Répartition de l\' importance de la neige sur les BV')
    repartition.ylabel('Nombre de BV')
    repartition.xlabel('Importance de la neige')
    repartition.hist(df.loc[df['seq_len'] == 7, 'NEI_F'], bins=binss)
    repartition.savefig(os.path.join(dir_plots, 'neige_population.png'))

    mae = plt.figure(figsize=fig_size)
    mae.title("MAE en fonction de l'importance de la neige")
    mae.ylabel('MAE')
    mae.xlabel('Importance de la neige')
    mae.scatter(df.loc[df['best_MAE'] == True, 'NEI_F'], df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    mae.scatter(df.loc[df['best_MAE'] == True, 'NEI_F'], df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    mae.legend()
    mae.grid(True)
    mae.savefig(os.path.join(dir_plots, 'neige_MAE.png'))

    nse = plt.figure(figsize=fig_size)
    nse.title("NSE en fonction de l'importance de la neige")
    nse.ylabel('NSE')
    mae.xlabel('Importance de la neige')
    nse.scatter(df.loc[df['best_NSE'] == True, 'NEI_F'], df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    nse.scatter(df.loc[df['best_NSE'] == True, 'NEI_F'], df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    nse.legend()
    nse.grid(True)
    nse.savefig(os.path.join(dir_plots, 'neige_NSE.png'))

# %%% Urbanisation

def plot_urbanisation(df):
    """Plot la répartition de l'urbanisation et la MAE/NSE en fonction de l'urbanisation"""
    keyword = 'DEV_' # 90,06,12

    mean_urb = df.loc[df['seq_len'] == 7, ['BV', 'DEV_90', 'DEV_06', 'DEV_12']].groupby('BV').mean()
    repartition = plt.figure(figsize=fig_size)
    repartition.title('Répartition de l\'urbanisation des BV')
    repartition.ylabel('Nombre de BV')
    repartition.xlabel('Urbanisation')
    repartition.hist(mean_urb, bins=binss)
    repartition.savefig(os.path.join(dir_plots, 'urbanisation_population.png'))

    urb_deriv_init = df.loc[df['seq_len'] == 7, ['BV', 'DEV_90', 'DEV_06', 'DEV_12']].groupby('BV')
    urb_deriv = urb_deriv_init.mean()

    repartition_deriv = plt.figure(figsize=fig_size)
    repartition_deriv.title('Répartition de l\'urbanisation des BV')
    repartition_deriv.ylabel('Nombre de BV')
    repartition_deriv.xlabel('vitesse d\'urbanisation')
    repartition_deriv.hist(urb_deriv, bins=binss)
    repartition_deriv.savefig(os.path.join(dir_plots, 'urbanisation_population_deriv.png'))




# trier par Q_m, seq_len, NEI_F, aridite(fait), ETPm, urbanisation, Pm, surface(fait)