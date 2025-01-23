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

    mae = plt.figure(figsize=fig_size)
    mae.title('MAE en fonction de l\'urbanisation')
    mae.ylabel('MAE')
    mae.xlabel('Urbanisation')
    mae.scatter(df.loc[df['best_MAE'] == True, keyword + '90'], df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    mae.scatter(df.loc[df['best_MAE'] == True, keyword + '90'], df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    mae.legend()
    mae.grid(True)
    mae.savefig(os.path.join(dir_plots, 'urbanisation_MAE.png'))

    nse = plt.figure(figsize=fig_size)
    nse.title('NSE en fonction de l\'urbanisation')
    nse.ylabel('NSE')
    nse.xlabel('Urbanisation')
    nse.scatter(df.loc[df['best_NSE'] == True, keyword + '90'], df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    nse.scatter(df.loc[df['best_NSE'] == True, keyword + '90'], df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    nse.legend()
    nse.grid(True)
    nse.savefig(os.path.join(dir_plots, 'urbanisation_NSE.png'))

    urb_deriv_init = df.loc[df['seq_len'] == 7, ['BV', 'DEV_90', 'DEV_06', 'DEV_12']].groupby('BV')
    urb_deriv = urb_deriv_init.mean()

    df['urbanisation_rate'] = (df[keyword + '12'] - df[keyword + '90']) / (2012 - 1990)

    repartition_deriv = plt.figure(figsize=fig_size)
    repartition_deriv.title('Répartition de l\'urbanisation des BV')
    repartition_deriv.ylabel('Nombre de BV')
    repartition_deriv.xlabel('vitesse d\'urbanisation')
    repartition_deriv.hist(df.loc[df['seq_len'] == 7, 'urbanisation_rate'], bins=binss)
    repartition_deriv.savefig(os.path.join(dir_plots, 'urbanisation_population_deriv.png'))


    mae_deriv = plt.figure(figsize=fig_size)
    mae_deriv.title('MAE en fonction de la variation d\'urbanisation')
    mae_deriv.ylabel('MAE')
    mae_deriv.xlabel('vitesse d\'urbanisation')
    mae_deriv.scatter(df.loc[df['best_MAE'] == True, 'urbanisation_rate'], df.loc[df['best_MAE'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    mae_deriv.scatter(df.loc[df['best_MAE'] == True, 'urbanisation_rate'], df.loc[df['best_MAE'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    mae_deriv.legend()
    mae_deriv.grid(True)
    mae_deriv.savefig(os.path.join(dir_plots, 'urbanisation_rate_MAE.png'))

    nse_deriv = plt.figure(figsize=fig_size)
    nse_deriv.title('NSE en fonction de la variation d\'urbanisation')
    nse_deriv.ylabel('NSE')
    nse_deriv.xlabel('vitesse d\'urbanisation')
    nse_deriv.scatter(df.loc[df['best_NSE'] == True, 'urbanisation_rate'], df.loc[df['best_NSE'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    nse_deriv.scatter(df.loc[df['best_NSE'] == True, 'urbanisation_rate'], df.loc[df['best_NSE'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    nse_deriv.legend()
    nse_deriv.grid(True)
    nse_deriv.savefig(os.path.join(dir_plots, 'urbanisation_rate_NSE.png'))




def plot_performance(df):
    """Plot la performance en NSE et MAE en fonction des aridités, urbanisations, surfaces"""
    # Group by aridite
    df['arid_group'] = pd.qcut(df['Arid'], 3, labels=['low', 'medium', 'high'])
    plot_boxplot(df, 'arid_group', 'Aridité')

    # Group by urbanisation
    df['urban_group'] = pd.qcut(df['DEV_12'], 3, labels=['low', 'medium', 'high'])
    plot_boxplot(df, 'urban_group', 'Urbanisation')

    # Group by surface
    df['surface_group'] = pd.qcut(df['AREA_SQKM_'], 3, labels=['small', 'medium', 'large'])
    plot_boxplot(df, 'surface_group', 'Surface')

def plot_boxplot(df, group_col, group_name):
    """Plot des boites à moustaches pour les groupes définis"""
    mae = plt.figure(figsize=fig_size)
    mae.title(f'MAE en fonction de {group_name}')
    mae.ylabel('MAE')
    df.boxplot(column='MAE_test_LSTM', by=group_col, ax=mae.gca())
    df.boxplot(column='MAE_test_GR4J', by=group_col, ax=mae.gca())
    mae.savefig(os.path.join(dir_plots, f'boxplot_MAE_{group_name}.png'))

    nse = plt.figure(figsize=fig_size)
    nse.title(f'NSE en fonction de {group_name}')
    nse.ylabel('NSE')
    df.boxplot(column='NSE_test_LSTM', by=group_col, ax=nse.gca())
    df.boxplot(column='NSE_test_GR4J', by=group_col, ax=nse.gca())
    nse.savefig(os.path.join(dir_plots, f'boxplot_NSE_{group_name}.png'))

# Plot 3D MAE/NSE en fonction de l'aridité, urbanisation
def plot_3D(df):
    """Plot 3D des MAE et NSE en fonction de l'aridité et de l'urbanisation"""
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Arid'], df['DEV_12'], df['MAE_test_LSTM'], c='red', label='MAE LSTM')
    ax.scatter(df['Arid'], df['DEV_12'], df['MAE_test_GR4J'], c='blue', label='MAE GR4J')
    ax.set_xlabel('Aridité')
    ax.set_ylabel('Urbanisation')
    ax.set_zlabel('MAE')
    ax.legend()
    fig.savefig(os.path.join(dir_plots, '3D_MAE.png'))

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(df['Arid'], df['DEV_12'], df['NSE_test_LSTM'], c='red', label='NSE LSTM')
    ax.scatter(df['Arid'], df['DEV_12'], df['NSE_test_GR4J'], c='blue', label='NSE GR4J')
    ax.set_xlabel('Aridité')
    ax.set_ylabel('Urbanisation')
    ax.set_zlabel('NSE')
    ax.legend()
    fig.savefig(os.path.join(dir_plots, '3D_NSE.png'))


# trier par Q_m, seq_len, NEI_F, aridite(fait), ETPm, urbanisation, Pm, surface(fait)

# %% Main
df = csv_df(fichier_resultat)

plot_moustaches(df)
plot_surfaces(df)
plot_aridite(df)
plot_neige(df)
plot_urbanisation(df)
plot_performance(df)
plot_3D(df)

# %% Fin