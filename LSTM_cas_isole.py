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
from LSTM import *

msg0 = '='*50
msg1 = '-'*50
print(msg0)

####################################################
# --------------- Entrées (modififiable) -----------

nom = 'Loic' # Fabio, Emma, Nicolas, Loic


# pour chaque BV, les "seq_len" suivants seront entrainés 
#(une ligne par combinaison BV-seq_len sur fichier sortie)
list_seq_len = [7, 15, 30]
loss_fonctions = ['MAE', 'NSE']


print('list_seq_len:', list_seq_len)

## getting the data repo and the list of files
dir_proj = os.path.normpath(os.getcwd())
print('dir_proj:', dir_proj) 

# fichier dir_data qui contient les données
dir_data = os.path.normpath(os.path.join(dir_proj, f'data_{nom}'))
print('dir_data:', dir_data)

# dossier résultast
dir_results = os.path.normpath(os.path.join(dir_proj, f'resultats'))
print('dir_results:', dir_results)

# fichier où le résultat va être enregistré
fichier_resultat = os.path.normpath(os.path.join(dir_results, f'resultats_LSTM_{nom}.csv'))
print('fichier_resultat:', fichier_resultat)

#####################################################

nom_BV = 'O0794010'
file_BV = f'{nom_BV}_tsdaily.csv'
print('file_BV:', file_BV)

seq_len = 7
loss_fonction = 'NSE'
LSTM_model = LSTM(dir_proj=dir_proj, dir_results=dir_results, loss_fonction=loss_fonction, file_BV=file_BV, seq_len=seq_len, nom=nom, verbose=1)
LSTM_model.load_data()
LSTM_model.preprocess_data()
LSTM_model.split_data()
LSTM_model.standardization()
LSTM_model.train()
LSTM_model.test_model()
LSTM_model.save_results(name=nom)