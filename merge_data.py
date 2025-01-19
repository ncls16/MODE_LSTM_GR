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

#####################################################
print(msg0)




resultats = os.listdir(dir_results)

# Création d'un DataFrame vide
df_resultats = pd.DataFrame()

# Loop through all result files containing 'LTSM' and add the tables to df_resultats
for file in resultats:
    if 'LTSM' in file:
        file_path = os.path.join(dir_results, file)
        df_temp = pd.read_csv(file_path)
        df_resultats = pd.concat([df_resultats, df_temp], ignore_index=True)

print(df_resultats)