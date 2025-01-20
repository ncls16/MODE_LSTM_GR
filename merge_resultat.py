import pandas as pd
import os

dir_proj = os.path.normpath(os.getcwd()) #.replace('\\','/') # Répertoire actuel
dir_results = os.path.normpath(os.path.join(dir_proj, 'resultats')) # Répertoire des résultats
name = ["Loic"] # Noms des fichiers #"Emma", "Loic", "Nicolas", "Fabio" 

file_GR4J = os.path.normpath(os.path.join(dir_results, 'resultats_GR4J.csv'))
df_GR4J = pd.read_csv(file_GR4J)
df_GR4J = df_GR4J.add_suffix('_GR4J')

file_stat = os.path.normpath(os.path.join(dir_results, 'statistiques.csv'))
df_stat = pd.read_csv(file_stat)

# Création d'un DataFrame vide
# best true or false
df_resultats = pd.DataFrame()
for file in name :
    file_LSTM = os.path.normpath(os.path.join(dir_results, "resultats_LSTM_" + file + ".csv"))
    df_LSTM = pd.read_csv(file_LSTM)
    df_resultats = pd.concat([df_resultats, df_LSTM], ignore_index=True)

df_resultats = df_resultats.add_suffix('_LSTM')
result = pd.merge(df_resultats, df_GR4J, how = 'left', left_on='BV_LSTM', right_on='BV_GR4J')   
result = pd.merge(result, df_stat, how = 'left', left_on='BV_LSTM', right_on='BV')
print(result)

# BEST
# code_BV = result['BV_LSTM'].unique()
# print(result.columns)
# for code in code_BV:
#     print(result[ ['BV_LSTM'] == code, ['seq_len'] == '7']['MAE_val_LSTM'])

# result['best'] = result['NSE_LSTM'] > result['NSE_GR4J']
# Sauvegarder le résultat dans un nouveau fichier CSV
# result.to_csv(dir_resultats + "resultats.csv", index=False)
# print("Le fichier 'resultats.csv' a été généré avec succès.")
