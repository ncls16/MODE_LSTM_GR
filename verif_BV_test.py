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
from GR4J import *

msg0 = '='*50
msg1 = '-'*50
print(msg0)

####################################################
# --------------- Entrées (modififiable) -----------

nom = 'Loic' # Fabio, Emma, Nicolas, Loic


# pour chaque BV, les "seq_len" suivants seront entrainés 
#(une ligne par combinaison BV-seq_len sur fichier sortie)
list_seq_len = [7]
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
print(msg0)



#----------------LSTM-----------------#
ts_files = os.listdir(dir_data)
file_BV = ts_files[-1] #Y5312010_tsdaily.csv
nom_BV = file_BV.split('_')[0]

# # Valeurs trouvées sur le code de M. SAADI :
# LSTM - NSE: 0.544, MAE (mm/d): 0.381 ==> Pour seq_len = 7
# GR4J - NSE: 0.883, MAE (mm/d): 0.244


# Essayer de retrouver les mêmes valeurs avec notre code
loss_fonction = 'NSE'
seq_len = 7
LSTM_model = LSTM(dir_proj=dir_proj, dir_results=dir_results, loss_fonction=loss_fonction, file_BV=file_BV, seq_len=seq_len, nom=nom, verbose=0)
LSTM_model.load_data()
LSTM_model.preprocess_data()
LSTM_model.split_data()
LSTM_model.standardization()
LSTM_model.train()
LSTM_model.test_model()
LSTM_model.save_results(name=nom)

#reproduire ce print : #LSTM - NSE: 0.452, MAE (mm/d): 0.427
print('LSTM')
print(f'résultat voulu en test: NSE: 0.452, MAE (mm/d): 0.427')
print(f'résultat obtenu : NSE: ')

print('BV	seq_len	ti_train	tf_train	ti_test	    tf_test	    NSE_train	MAE_train	NSE_val	    MAE_val	    NSE_test	MAE_test	training_finished	epoch	loss_func   training_time	    nom')
print('Y5312010	7	1997-01-07	2004-12-05	2008-03-31	2012-12-31	0.6178586	0.27755117	-0.32663345	0.29669937	0.4346555	0.42474118	True	            99	    NSE 	    20.1180686950684	Loic')

print(msg1)

#----------------GR4J-----------------#

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
df_GR4J = gr4j_model.save_results()

# reproduire ce print : GR4J - NSE: 0.883, MAE (mm/d): 0.244
print('résultat voulu : NSE: 0.883, MAE (mm/d): 0.244')
print(f'GR4J - NSE: {df_GR4J.loc[0, "NSE_test"]}, MAE (mm/d): {df_GR4J.loc[0, "MAE_test"]}')
# obtenu : GR4J - NSE: 0.8282378839052247, MAE (mm/d): 0.2087258551241016


