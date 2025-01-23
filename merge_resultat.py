import pandas as pd
import os
import numpy as np


####################################################
# --------------- Entrées (modififiable) -----------
dir_proj = os.path.normpath(os.getcwd()) 
dir_results = os.path.normpath(os.path.join(dir_proj, 'resultats')) # Répertoire des résultats
#name = ["Loic"] # Noms des fichiers #"Emma", "Loic", "Nicolas", "Fabio" 

# dossier data
dir_data = os.path.normpath(os.path.join(dir_proj, 'data'))

# fichier résultat GR4J
file_GR4J = os.path.normpath(os.path.join(dir_results, 'resultats_GR4J.csv'))
df_GR4J = pd.read_csv(file_GR4J)
df_GR4J = df_GR4J.add_suffix('_GR4J')

# statistiques par BV
file_stat = os.path.normpath(os.path.join(dir_results, 'statistiques.csv'))
df_stat = pd.read_csv(file_stat)

#####################################################



# récupération des fichiers de résultats
resultats = os.listdir(dir_results)

# Récupération données LSTM
df_LSTM = pd.DataFrame()
resultats = os.listdir(dir_results)
print('fichiers résultats : ', resultats)
for file in resultats :
    if 'LSTM' in file:
        file_path = os.path.join(dir_results, file)
        df_temp = pd.read_csv(file_path)
        df_LSTM = pd.concat([df_LSTM, df_temp], ignore_index=True)

df_LSTM = df_LSTM.add_suffix('_LSTM')


# # ----------------------Merge des données---------------------
# merge LSTM + GR4J
df_result = pd.merge(df_LSTM, df_GR4J, how = 'left', left_on='BV_LSTM', right_on='BV_GR4J')

# merge avec les statistiques   
df_result = pd.merge(df_result, df_stat, how = 'left', left_on='BV_LSTM', right_on='BV')

print('df_result.shape : ',df_result.shape)





# -----------------------Normalisation du temps de calcul---------------------
noms = df_result['nom_LSTM'].unique()
print('df_result.columns : ',df_result.columns)
print('noms = ',noms)


liste_BV_reference = liste_BV_reference = [filename.split('_')[0] for filename in os.listdir(dir_data)[:5]]
print('BV_reference : ',liste_BV_reference)
df_result_ref = df_result[df_result['BV_LSTM'].isin(liste_BV_reference)]

#df_result['training_time_LSTM'].groupby('nom_LSTM').mean()
mean_training_times_ref = df_result_ref.groupby('nom_LSTM')['training_time_LSTM'].mean()
min_training_time_name = mean_training_times_ref.idxmin() #personne qui a la moyenne d'entrainement la plus faible
min_training_time_value = mean_training_times_ref.min()


# Normaliser le temps d'entraînement par rapport à la personne qui a la moyenne la plus faible
df_result['training_time_norm_LSTM'] = (
    df_result['training_time_LSTM'] *
    min_training_time_value / 
    df_result['nom_LSTM'].map(mean_training_times_ref)
)



# BEST
# code_BV = result['BV_LSTM'].unique()
# print(result.columns)
# for code in code_BV:
#     print(result[ ['BV_LSTM'] == code, ['seq_len'] == '7']['MAE_val_LSTM'])

# result['best'] = result['NSE_LSTM'] > result['NSE_GR4J']
# Sauvegarder le résultat dans un nouveau fichier CSV
# result.to_csv(dir_resultats + "resultats.csv", index=False)
# print("Le fichier 'resultats.csv' a été généré avec succès.")
