## libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import os, re, time, sys
from tqdm.notebook import tqdm
import time
import spotpy

# import des fonctions dans les autres fichiers
from fonction import *
from LTSM import *

msg0 = '='*50
msg1 = '-'*50
print(msg0)

####################################################
# --------------- Entrées (modififiable) -----------

nom = 'Loic' # Fabio, Emma, Nicolas, Loic


# pour chaque BV, les "seq_len" suivants seront entrainés 
#(une ligne par combinaison BV-seq_len sur fichier sortie)
list_seq_len = [7, 15, 30]
print('list_seq_len:', list_seq_len)

## getting the data repo and the list of files
dir_proj = os.path.normpath(os.getcwd())
print('dir_proj:', dir_proj) # si pas le bon, modifier avec : dir_proj = ...

# fichier dir_data qui contient les données
dir_data = os.path.normpath(os.path.join(dir_proj, f'data_{nom}'))
print('dir_data:', dir_data)

# dossier résultast
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
print('dir_results:', dir_results)

# fichier où le résultat va être enregistré
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'resultats_LTSM_{nom}.csv'))
print('fichier_resultat:', fichier_resultat)

# fichier d'observations # (pas sur de l'utilité de obs_t)
obs_file = os.path.normpath(os.path.join(dir_data, 'obs_t.csv'))
print('obs_file:', obs_file)

#####################################################
print(msg0)


print('Début du traitement à :', time.ctime())



#------------------Recuperation Data------------------#
ts_files = os.listdir(dir_data)
ts_files = [file for file in ts_files if file.endswith(".csv")] # liste de tous les BV
obs_t = np.loadtxt(obs_file, delimiter = ',')

print('ts_files[:10]:\n', ts_files[:10])
print('ts_files à traiter : ', len(ts_files))
print(msg1)




#boucle d'entrainement
for file_BV in tqdm(ts_files, desc='BV', ncols=100,ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {rate_fmt}') :
    for seq_len in list_seq_len :
        
        # verification si (BV-seq_len) a déjà été avec succès
        if os.path.exists(fichier_resultat):
            df = pd.read_csv(fichier_resultat)
            print('df.columns:', df.columns)
            print('df = ',df)
            if not df[(df['BV'] == file_BV) & (df['seq_len'] == seq_len)].empty and df['training_finished'] == True:
                print(f"BV {file_BV} avec seq_len {seq_len} déjà traité")
                continue
            
            elif not df[(df['BV'] == file_BV) & (df['seq_len'] == seq_len)].empty and df['training_finished'] == False:
                # retirer la ligne si le training n'a pas été un succès
                df = df[~((df['BV'] == file_BV) & (df['seq_len'] == seq_len))] 
                df.to_csv(fichier_resultat, index=False)
        
        
        # Modèle LTSM
        LTSM_model = LTSM(dir_proj=dir_proj, dir_results=dir_results, file_BV=file_BV, obs_t=obs_t, seq_len=seq_len, verbose=1)
        LTSM_model.load_data()
        LTSM_model.preprocess_data()
        LTSM_model.split_data()
        LTSM_model.standardization()
        LTSM_model.train()
        LTSM_model.test_model()
        LTSM_model.save_results(name=nom)

