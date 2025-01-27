# %% Librairies
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

# %% Variables
fig_size = (10, 5) # taille des figures
binss = 20 # nombre de bins pour les histogrammes
# dossier origine
dir_proj = os.path.normpath(os.getcwd())
# dossier resultat
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
# print('dir_results:', dir_results)

# fichier total
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'fusion_shp_resultats.csv'))
if not os.path.exists(fichier_resultat):
    print(f'Le fichier {fichier_resultat} n\'existe pas')

# Dossier ou sont stockés les plots
dir_plots = os.path.normpath(os.path.join(dir_proj, f'plots'))
os.makedirs(dir_plots, exist_ok=True)


# %% Fonctions -----------------------------------------------------------------------------
def csv_df(file: str) -> pd.DataFrame:
    """Convertir un fichier csv en DataFrame"""
    df = pd.read_csv(file)
    # list_columns_to_expect =
    # df = df.loc[:, df.columns.str.contains('BV|MAE test|NSE test|epoch|seq_len')]

    return df

# %% Affichage des données -----------------------------------------------------------------
# %%% seq_len
def plot_moustaches(df):
    """Plot la distribution des MAE et NSE"""
    seq_len = df['seq_len_LSTM'].unique()

    timed, ax = plt.subplots(figsize=(10, 6))  # Créez une figure et des axes
    timed.suptitle('Distribution des temps de calcul')  # Utilisez suptitle pour ajouter un titre à la figure
    ax.set_ylabel('Temps de calcul (s)')  # Utilisez set_ylabel pour définir l'étiquette de l'axe y
    ax.boxplot(df.loc[df['best_MAE_val_LSTM'] == True, 'temps_calibration_GR4J'], positions = [1], tick_labels=['GR4J'], showfliers=False)
    ax.boxplot(df.loc[df['best_MAE_val_LSTM'] == True, 'training_time_norm_LSTM'], positions = [2], tick_labels=['MAE_LSTM'], showfliers=False)
    ax.boxplot(df.loc[df['best_NSE_val_LSTM'] == True, 'training_time_norm_LSTM'], positions = [3], tick_labels=['NSE_LSTM'], showfliers=False)
    ax.set_yscale('log')  # Utilisez set_yscale pour définir l'échelle de l'axe y

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle(f'Distribution des MAE pour différents seq_len')
    ax1.set_ylabel('MAE')
    ax1.boxplot(df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], positions = [1], tick_labels=['GR4J'], showfliers=False)
    ax1.boxplot(df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], positions = [2], tick_labels=['LSTM_best_MAE'], showfliers=False)

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle(f'Distribution des NSE pour différents seq_len')
    ax2.set_ylabel('NSE')
    ax2.boxplot(df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], positions = [1], tick_labels=['GR4J'], showfliers=False)
    ax2.boxplot(df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], positions = [2], tick_labels=['LSTM_best_NSE'], showfliers=False)


    pos = 3
    for sl in seq_len:
        ax.boxplot(df.loc[df['seq_len_LSTM'] == sl, 'training_time_norm_LSTM'], positions = [pos+1], tick_labels=[f'LSTM_{sl}'], showfliers=False)
        ax1.boxplot(df.loc[df['seq_len_LSTM'] == sl, 'MAE_test_LSTM'], positions = [pos], tick_labels=[f'LSTM_MAE_{sl}'], showfliers=False)
        ax2.boxplot(df.loc[df['seq_len_LSTM'] == sl, 'NSE_test_LSTM'], positions = [pos], tick_labels=[f'LSTM_NSE_{sl}'], showfliers=False)
        pos += 1

    timed.savefig(os.path.join(dir_plots, 'boxplot_time.png'))
    mae.savefig(os.path.join(dir_plots, 'boxplot_MAE.png'))
    nse.savefig(os.path.join(dir_plots, 'boxplot_NSE.png'))
    plt.close(timed)
    plt.close(mae)
    plt.close(nse)

# log en nse
# 

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
    repartition, ax = plt.subplots(figsize=fig_size)
    repartition.suptitle('Répartition des surfaces')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Surface (km²)')
    ax.hist(df.loc[df['seq_len_LSTM'] == 7, 'AREA_SQKM_'], bins=binss)
    ax.set_xscale('log')
    repartition.savefig(os.path.join(dir_plots, 'surfaces_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de la surface')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Surface (km²)')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'AREA_SQKM_'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'AREA_SQKM_'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    mae.savefig(os.path.join(dir_plots, 'surfaces_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de la surface')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Surface (km²)')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'AREA_SQKM_'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'AREA_SQKM_'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'surfaces_NSE.png'))
    plt.close(repartition)
    plt.close(mae)
    plt.close(nse)

# %%% Aridite potentielle
def plot_aridite(df):
    """Plot la répartition de l'aridité et la MAE/NSE en fonction de l'aridité"""
    aridite, ax = plt.subplots(figsize=fig_size)
    aridite.suptitle('Répartition de l\'aridité')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Aridité')
    ax.hist(df.loc[df['seq_len_LSTM'] == 7, 'Arid'], bins=binss)
    aridite.savefig(os.path.join(dir_plots, 'aridite_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de l\'aridité')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Aridité')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'Arid'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'Arid'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    ax1.set_xscale('log')
    mae.savefig(os.path.join(dir_plots, 'aridite_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de l\'aridité')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Aridité')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'Arid'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'Arid'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.set_xscale('log')
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'aridite_NSE.png'))
    
    plt.close(aridite)
    plt.close(mae)
    plt.close(nse)

# %%% Debit
def plot_debit(df):
    """Plot la répartition du débit et la MAE/NSE en fonction du débit"""

    data_ax = df.loc[df['best_MAE_val_LSTM'] == True, 'mean_flow_mm'] / df.loc[df['best_MAE_val_LSTM'] == True, 'mean_precip']

    debit, ax = plt.subplots(figsize=(10,6))
    debit.suptitle('Répartition du débit')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Coefficient de Ruissellement, $\dfrac{Débit_m}{Précipitations_m}$')
    ax.hist(data_ax, bins=binss)
    # ax.hist(df.loc[df['seq_len_LSTM'] == 7, 'mean_flow_mm'], bins=binss)
    debit.savefig(os.path.join(dir_plots, 'ruissellement_population.png'))

    mae, ax1 = plt.subplots(figsize=(10,6))
    mae.suptitle('MAE en fonction du ruissellement')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Coefficient de ruissellement, $\dfrac{Débit_m}{Précipitations_m}$')
    ax1.scatter(data_ax, df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(data_ax, df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    ax1.grid(True, which='both')
    mae.savefig(os.path.join(dir_plots, 'ruissellement_MAE.png'))

    nse, ax2 = plt.subplots(figsize=(10,6))
    nse.suptitle('NSE en fonction du débit')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Coefficient de ruissellement, $\dfrac{Débit_m}{Précipitations_m}$')
    ax2.scatter(data_ax, df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(data_ax, df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.grid(True, which='both')
    nse.savefig(os.path.join(dir_plots, 'ruissellement_NSE.png'))

    plt.close(debit)
    plt.close(mae)
    plt.close(nse)

# %%% Neige
def plot_neige(df):
    """Plot la répartition de la neige et la MAE/NSE en fonction de la neige"""
    neige, ax = plt.subplots(figsize=fig_size)
    neige.suptitle('Répartition de la neige')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Importance de la neige (%)')
    ax.hist(df.loc[df['seq_len_LSTM'] == 7, 'NEI_F'], bins=binss)
    neige.savefig(os.path.join(dir_plots, 'neige_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de l\'importance de la neige')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Importance de la neige (%)')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'NEI_F'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'NEI_F'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    mae.savefig(os.path.join(dir_plots, 'neige_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de l\'importance de la neige')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Importance de la neige (%)')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'NEI_F'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'NEI_F'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'neige_NSE.png'))

    plt.close(neige)
    plt.close(mae)
    plt.close(nse)

# %% ETP
def plot_etp(df):
    """Plot la répartition de l'etp et la MAE/NSE en fonction de l'etp"""
    etp, ax = plt.subplots(figsize=fig_size)
    etp.suptitle('Répartition de l\'ETP')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('ETP (mm)')
    ax.hist(df.loc[df['best_MAE_val_LSTM'] == 7, 'ETPm'], bins=binss)
    etp.savefig(os.path.join(dir_plots, 'etp_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de l\'ETP')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('ETP (mm)')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'ETPm'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'ETPm'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    mae.savefig(os.path.join(dir_plots, 'etp_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de l\'ETP')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('ETP (mm)')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'ETPm'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'ETPm'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'etp_NSE.png'))

    plt.close(etp)
    plt.close(mae)
    plt.close(nse)

# %%% plot pluie
def plot_pluie(df):
    """Plot la répartition de la pluie et la MAE/NSE en fonction de la pluie"""
    pluie, ax = plt.subplots(figsize=fig_size)
    pluie.suptitle('Répartition de la pluie')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Pluie (mm)')
    ax.hist(df.loc[df['seq_len_LSTM'] == 7, 'Pm'], bins=binss)
    pluie.savefig(os.path.join(dir_plots, 'pluie_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de la pluie')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Pluie (mm)')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'Pm'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'Pm'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    mae.savefig(os.path.join(dir_plots, 'pluie_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de la pluie')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Pluie (mm)')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'Pm'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'Pm'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'pluie_NSE.png'))

    plt.close(pluie)
    plt.close(mae)
    plt.close(nse)

# %%% Urbanisation
def plot_urbanisation(df):
    """Plot la répartition de l'urbanisation et la MAE/NSE en fonction de l'urbanisation"""
    keyword = 'DEV'  # 90,06,12

    mean_urb = df.loc[df['best_MAE_val_LSTM'] == True, ['BV', 'DEV90', 'DEV00', 'DEV06', 'DEV12']].groupby('BV').mean()
    repartition, ax = plt.subplots(figsize=fig_size)
    repartition.suptitle('Répartition de l\'urbanisation des BV')
    ax.set_ylabel('Nombre de BV')
    ax.set_xlabel('Urbanisation')
    ax.hist(mean_urb, bins=binss, label=['1990', '2000', '2006', '2012'])
    ax.legend()
    repartition.savefig(os.path.join(dir_plots, 'urbanisation_population.png'))

    mae, ax1 = plt.subplots(figsize=fig_size)
    mae.suptitle('MAE en fonction de l\'urbanisation')
    ax1.set_ylabel('MAE')
    ax1.set_xlabel('Urbanisation (%)')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, keyword + '90'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax1.scatter(df.loc[df['best_MAE_val_LSTM'] == True, keyword + '90'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax1.legend()
    ax1.grid(True)
    mae.savefig(os.path.join(dir_plots, 'urbanisation_MAE.png'))

    nse, ax2 = plt.subplots(figsize=fig_size)
    nse.suptitle('NSE en fonction de l\'urbanisation')
    ax2.set_ylabel('NSE')
    ax2.set_xlabel('Urbanisation (%)')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, keyword + '90'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax2.scatter(df.loc[df['best_NSE_val_LSTM'] == True, keyword + '90'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax2.legend()
    ax2.grid(True)
    nse.savefig(os.path.join(dir_plots, 'urbanisation_NSE.png'))

    df['urbanisation_rate'] = (df[keyword + '12'] - df[keyword + '90']) / (2012 - 1990)

    repartition_deriv, ax3 = plt.subplots(figsize=fig_size)
    repartition_deriv.suptitle('Répartition de l\'urbanisation des BV')
    ax3.set_ylabel('Nombre de BV')
    ax3.set_xlabel('Vitesse d\'urbanisation')
    ax3.hist(df.loc[df['seq_len_LSTM'] == 7, 'urbanisation_rate'], bins=binss)
    repartition_deriv.savefig(os.path.join(dir_plots, 'urbanisation_population_deriv.png'))

    mae_deriv, ax4 = plt.subplots(figsize=fig_size)
    mae_deriv.suptitle('MAE en fonction de la variation d\'urbanisation')
    ax4.set_ylabel('MAE')
    ax4.set_xlabel('Vitesse d\'urbanisation')
    ax4.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'urbanisation_rate'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], label='LSTM', color='red')
    ax4.scatter(df.loc[df['best_MAE_val_LSTM'] == True, 'urbanisation_rate'], df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_GR4J'], label='GR4J', color='blue')
    ax4.legend()
    ax4.grid(True)
    mae_deriv.savefig(os.path.join(dir_plots, 'urbanisation_rate_MAE.png'))

    nse_deriv, ax5 = plt.subplots(figsize=fig_size)
    nse_deriv.suptitle('NSE en fonction de la variation d\'urbanisation')
    ax5.set_ylabel('NSE')
    ax5.set_xlabel('Vitesse d\'urbanisation')
    ax5.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'urbanisation_rate'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_LSTM'], label='LSTM', color='red')
    ax5.scatter(df.loc[df['best_NSE_val_LSTM'] == True, 'urbanisation_rate'], df.loc[df['best_NSE_val_LSTM'] == True, 'NSE_test_GR4J'], label='GR4J', color='blue')
    ax5.legend()
    ax5.grid(True)
    nse_deriv.savefig(os.path.join(dir_plots, 'urbanisation_rate_NSE.png'))

    plt.close(repartition)
    plt.close(mae)
    plt.close(nse)


def plot_performance(df):
    """Plot la performance en NSE et MAE en fonction des aridités, urbanisations, surfaces"""
    labelss = ['low', 'medium', 'high']
    # Group by aridite
    df['arid_group'] = pd.qcut(df['Arid'], len(labelss), labels=labelss)
    plot_boxplot(df, 'arid_group', 'Aridité', labelss)

    # Group by urbanisation
    df['urban_group'] = pd.qcut(df['DEV12'], len(labelss), labels=labelss)
    plot_boxplot(df, 'urban_group', 'Urbanisation', labelss)

    # Group by surface
    df['surface_group'] = pd.qcut(df['AREA_SQKM_'], len(labelss), labels=labelss)
    plot_boxplot(df, 'surface_group', 'Surface', labelss)

    # Group by neige
    df['neige_group'] = pd.qcut(df['NEI_F'], len(labelss), labels=labelss)
    plot_boxplot(df, 'neige_group', 'Neige', labelss)

    # Group by etp
    df['etp_group'] = pd.qcut(df['ETPm'], len(labelss), labels=labelss)
    plot_boxplot(df, 'etp_group', 'ETP', labelss)

    # Group by pluie
    df['pluie_group'] = pd.qcut(df['Pm'], len(labelss), labels=labelss)
    plot_boxplot(df, 'pluie_group', 'Pluie', labelss)

    # Group by ruissellement
    df['ruissellement_group'] = pd.qcut(df['mean_flow_mm'] / df['mean_precip'], len(labelss), labels=labelss)
    plot_boxplot(df, 'ruissellement_group', 'Ruissellement', labelss)

def plot_boxplot(df, group_col, group_name, labels):
    """Plot des boites à moustaches pour les groupes définis"""
    
    # Création du premier plot pour MAE
    mae, ax1 = plt.subplots(figsize=(10, 6))
    mae.suptitle(f'MAE en fonction de {group_name}')
    ax1.set_ylabel('MAE')
    
    pos = 1
    for type in labels:
        if type !=labels[-1]:
            ax1.axvline(x=pos + 0.5, color='grey', linestyle='--')

        data1 = df.loc[df[group_col] == type, 'MAE_test_LSTM']
        ax1.boxplot(data1, positions=[pos - 0.2], tick_labels=['LSTM'])
        data2 = df.loc[df[group_col] == type, 'MAE_test_GR4J']
        ax1.boxplot(data2, positions=[pos + 0.2], tick_labels=['GR4J'])
        pos += 1

    mae.savefig(os.path.join(dir_plots, f'boxplot_MAE_{group_name}.png'))

    # Création du deuxième plot pour NSE
    nse, ax2 = plt.subplots(figsize=(10, 6))
    nse.suptitle(f'NSE en fonction de {group_name}')
    ax2.set_ylabel('NSE')
    
    pos = 1
    for type in labels:
        if type !=labels[-1]:
            ax2.axvline(x=pos + 0.5, color='grey', linestyle='--')

        data1 = df.loc[df[group_col] == type, 'NSE_test_LSTM']
        ax2.boxplot(data1, positions=[pos - 0.2], tick_labels=['LSTM'])
        data2 = df.loc[df[group_col] == type, 'NSE_test_GR4J']
        ax2.boxplot(data2, positions=[pos + 0.2], tick_labels=['GR4J'])
        pos += 1

    nse.savefig(os.path.join(dir_plots, f'boxplot_NSE_{group_name}.png'))

    plt.close(mae)
    plt.close(nse)


def boxplot_mae_nse(df):
    """Plot des boites à moustaches pour les MAE et NSE"""
    mae, ax1 = plt.subplots(figsize=(10, 6))
    mae.suptitle('MAE en fonction de l\'optimisation des MAE et NSE')
    ax1.set_ylabel('MAE')
    ax1.boxplot(df.loc[df['best_MAE_val_LSTM'] == True, 'MAE_test_LSTM'], positions=[1], tick_labels=['MAE'])
    ax1.boxplot(df.loc[df['best_NSE_val_LSTM'] == True, 'MAE_test_LSTM'], positions=[2], tick_labels=['NSE'])
    mae.savefig(os.path.join(dir_plots, 'boxplot_MAE_NSE.png'))


# Plot 3D MAE/NSE en fonction de l'aridité, urbanisation
def plot_3D(df):
    """Plot 3D des MAE et NSE en fonction de l'aridité et de l'urbanisation"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10), subplot_kw={'projection': '3d'})

    ax1.scatter(df['Arid'], df['DEV12'], df['MAE_test_LSTM'], c='red', label='MAE LSTM')
    ax1.scatter(df['Arid'], df['DEV12'], df['MAE_test_GR4J'], c='blue', label='MAE GR4J')
    ax1.set_xlabel('Aridité')
    ax1.set_ylabel('Urbanisation')
    ax1.set_zlabel('MAE')
    ax1.legend()
    ax1.set_title('3D MAE')

    ax2.scatter(df['Arid'], df['DEV12'], df['NSE_test_LSTM'], c='red', label='NSE LSTM')
    ax2.scatter(df['Arid'], df['DEV12'], df['NSE_test_GR4J'], c='blue', label='NSE GR4J')
    ax2.set_xlabel('Aridité')
    ax2.set_ylabel('Urbanisation')
    ax2.set_zlabel('NSE')
    ax2.legend()
    ax2.set_title('3D NSE')

    fig.savefig(os.path.join(dir_plots, '3D_MAE_NSE.png'))

    plt.close(fig)

# trier par Q_m, seq_len, NEI_F, aridite(fait), ETPm, urbanisation, Pm, surface(fait)

# %% Affichage des données
def affichage_debit_pluie(df):
    return True

def affichage_debit_aridite(df):
    return True

def affichage_debit_etp(df):
    return True

# %% Main
if __name__ == '__main__':
    # Chargement des données
    df = csv_df(fichier_resultat)
    plot_moustaches(df)
    plot_surfaces(df)
    plot_aridite(df)
    plot_debit(df)
    plot_neige(df)
    plot_etp(df)
    plot_pluie(df)
    plot_urbanisation(df)
    plot_performance(df)
    boxplot_mae_nse(df)
    plot_3D(df)
    print(f'Plots sauvegardés dans le dossier plots : {dir_plots}')

# %% Fin