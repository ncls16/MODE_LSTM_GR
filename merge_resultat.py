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

# Best MAE and NSE in fucntion of seq_len
code_BV = result['BV_LSTM'].unique()
for code in code_BV:
    MAE = result.loc[result['BV_LSTM'] == code,['seq_len_LSTM', 'MAE_val_LSTM']] 
    MAE_max_seq_len = MAE.loc[MAE['MAE_val_LSTM'].idxmax(), 'seq_len_LSTM']
    result.loc[(result['BV_LSTM'] == code) & (result['seq_len_LSTM'] == MAE_max_seq_len),'best_MAE_val_LSTM'] = True

    NSE = result.loc[result['BV_LSTM'] == code,['seq_len_LSTM', 'NSE_val_LSTM']]
    NSE_max_seq_LSTM = NSE.loc[NSE['NSE_val_LSTM'].idxmax(), 'seq_len_LSTM']
    result.loc[(result['BV_LSTM'] == code) & (result['seq_len_LSTM'] == NSE_max_seq_LSTM),'best_NSE_val_LSTM'] = True

result['best_MAE_val_LSTM'] = result['best_MAE_val_LSTM'].fillna(False)
result['best_NSE_val_LSTM'] = result['best_NSE_val_LSTM'].fillna(False)

# Sauvegarder le résultat dans un nouveau fichier CSV
result.to_csv(dir_results + "resultats.csv", index=False)
print("Le fichier 'resultats.csv' a été généré avec succès.")
