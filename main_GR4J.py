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
from GR4J import *

msg0 = '='*50
msg1 = '-'*50
print(msg0)

####################################################
# --------------- Entrées (modififiable) -----------

## getting the data repo and the list of files
dir_proj = os.path.normpath(os.getcwd())
print('dir_proj:', dir_proj) # si pas le bon, modifier avec : dir_proj = ...

# fichier dir_data qui contient les données
dir_data = os.path.normpath(os.path.join(dir_proj, f'data'))
if not os.path.exists(dir_data):
    os.mkdir(dir_data)
print('dir_data:', dir_data)

# dossier résultast
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
if not os.path.exists(dir_results):
    os.mkdir(dir_results)
print('dir_results:', dir_results)

# fichier où le résultat va être enregistré
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'resultats_GR4J.csv'))
print('fichier_resultat:', fichier_resultat)

# # fichier d'observations # (pas sur de l'utilité de obs_t)
# obs_file = os.path.normpath(os.path.join(dir_data, 'obs_t.csv'))
# print('obs_file:', obs_file)

#####################################################
print(msg0)


print('Début du traitement à :', time.ctime())



#------------------Recuperation Data------------------#
ts_files = os.listdir(dir_data)
ts_files = [file for file in ts_files if file.endswith(".csv")] # liste de tous les BV

print('ts_files[:10]:\n', ts_files[:10])
print('ts_files à traiter : ', len(ts_files))
print(msg1)



#boucle d'entrainement
for file_BV in tqdm(ts_files, desc='BV', ncols=100, ascii=True, bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} {elapsed}<{remaining} {rate_fmt}') :
    nom_BV = file_BV.split('_')[0]
        
    # verification si (BV-seq_len) a déjà été avec succès
    if os.path.exists(fichier_resultat):
        df = pd.read_csv(fichier_resultat)
        
        if not df[(df['BV'] == nom_BV)].empty and df['training_finished'] == True:
            print(f"BV {file_BV} déjà traité")
            continue
        
        elif not df[(df['BV'] == nom_BV)].empty and df['training_finished'] == False:
            # retirer la ligne si le training n'a pas été un succès
            df = df[~((df['BV'] == nom_BV) )] 
            df.to_csv(fichier_resultat, index=False)
            
    if not file_BV.endswith(".csv"):
        print(f"Le fichier {file_BV} n'est pas un fichier .csv")
        continue
    elif "obs_t" in file_BV:
        print(f"Le fichier {file_BV} est un fichier d'observations")
        continue
    
    
    ## Modèle GR4J
    print(f"Traitement du BV : {file_BV}")
    gr4j_model = GR4JModel(dir_proj=dir_proj, file_name=file_BV, verbose=0)
    gr4j_model.load_data()
    gr4j_model.preprocess_data()
    gr4j_model.split_data()

    # Paramètres pour la calibration
    msk_optim = np.array([True, True, True, True])
    val_nonoptim = np.ones(4) * np.nan
    StartParamDistribT = np.array([[5.13, -1.60, 3.03, -9.05],
                                    [5.51, -0.61, 3.74, -8.51],
                                    [6.07, -0.02, 4.42, -8.06]])


    param_final, crit_final = gr4j_model.calibrate_model(
                                run_model=run_GR4J,
                                msk_optim=msk_optim,
                                val_nonoptim=val_nonoptim,
                                StartParamDistribT=StartParamDistribT,
                                loss_func=RMSE
                            )
    gr4j_model.test_model(run_GR4J)
    gr4j_model.save_results()