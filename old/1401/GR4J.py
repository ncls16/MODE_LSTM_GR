## libraries
import matplotlib.pyplot as plt
from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import os, re, time, sys
from tqdm.notebook import tqdm
from hydrogr import InputDataHandler, ModelGr4j
import spotpy
import time
from fonction import *

msg0 = '='*50
msg1 = '-'*50

#------------------Recuperation Data------------------#
## getting the data repo and the list of files
dir_proj = re.sub("\\\\","/", os.getcwd())
print('dir_proj:', dir_proj)
dir_data = f"{dir_proj}/data"
ts_files = os.listdir(dir_data)
ts_files = [file for file in ts_files if file.endswith(".csv")]

obs_file = os.path.join(dir_data, 'obs_t.csv')
obs_t = np.loadtxt(obs_file, delimiter = ',')

print(obs_t)
# print(msg0)
# print("List of files in the data directory:")
# print(ts_files) 
# print(msg0)




class GR4JModel:
    def __init__(self, dir_proj, file_name, seq_len=30, tr_p=0.5, val_p=0.2, test_p=0.3, verbose=0):
        """
        Initialise la classe pour un seul bassin versant.

        :param dir_data: Répertoire contenant les fichiers de données.
        :param file_name: Nom du fichier pour le bassin versant.
        :param seq_len: Longueur de la séquence pour le modèle.
        :param tr_p: Proportion des données pour l'entraînement.
        :param val_p: Proportion des données pour la validation.
        :param test_p: Proportion des données pour le test.
        """
        self.dir_proj   = dir_proj
        self.dir_data   = os.path.join(dir_proj, "data")
        self.file_name  = file_name
        self.seq_len    = seq_len
        self.tr_p       = tr_p
        self.val_p      = val_p
        self.test_p     = test_p
        self.data_cat   = None
        self.date_rshp_ = None
        self.x_rshp_    = None
        self.y_rshp_    = None
        self.verbose    = verbose
        
        colonnes_resultats = ["BV", "param_final", "temps_calibration", "NSE_calibration", "ti_train", "tf_train", "ti_test", "tf_test", "NSE_test", "MAE_test", "training_finished"]
        self.dic_resultats = {col: None for col in colonnes_resultats}

    def load_data(self):
        """Charge les données pour le bassin versant spécifié."""
        file_path = f"{self.dir_data}/{self.file_name}"
        self.data_cat = pd.read_csv(file_path, sep=";")
        self.data_cat.columns = ['date', 'precipitation', 'evapotranspiration', 'flow_mm']
        self.data_cat.index = pd.to_datetime(self.data_cat.date)
        
        if self.verbose == 1:
            print("Data loaded for basin:", self.file_name)

    def preprocess_data(self):
        """Prétraite les données en créant les entrées et sorties pour le modèle."""
        input_feat = ["precipitation", "evapotranspiration"]
        output_feat = ["flow_mm"]
        
        x_in = np.array(self.data_cat.drop([*output_feat, "date"], axis=1))
        y_out = np.array(self.data_cat[output_feat])
        x_rshp, y_rshp = reshape_data(x=x_in, y=y_out, seq_length=self.seq_len)
        date_rshp = self.data_cat.date[(self.seq_len - 1):].values

        # Supprime les cas avec des NaN dans les sorties
        y_nan = np.where(np.isnan(y_rshp), 1, 0)
        y_nan_ = np.sum(y_nan, axis=1)
        self.date_rshp_ = date_rshp[y_nan_ == 0]
        self.x_rshp_ = x_rshp[y_nan_ == 0, ...]
        self.y_rshp_ = y_rshp[y_nan_ == 0, ...]
        
        if self.verbose == 1:
            print("Data preprocessed. Shapes:", self.x_rshp_.shape, self.y_rshp_.shape)

    def split_data(self):
        """Sépare les données en ensembles d'entraînement, de validation et de test."""
        n_samples = self.date_rshp_.size
        tr_smpl = int(self.tr_p * n_samples)
        tr_val_smpl = int((self.tr_p + self.val_p) * n_samples)

        # training
        self.date_train = self.date_rshp_[:tr_smpl]
        self.x_train = self.x_rshp_[:tr_smpl, ...]
        self.y_train = self.y_rshp_[:tr_smpl, ...]

        # validation
        self.date_val = self.date_rshp_[tr_smpl:tr_val_smpl]
        self.x_val = self.x_rshp_[tr_smpl:tr_val_smpl, ...]
        self.y_val = self.y_rshp_[tr_smpl:tr_val_smpl, ...]

        # test
        self.date_test = self.date_rshp_[tr_val_smpl:]
        self.x_test = self.x_rshp_[tr_val_smpl:, ...]
        self.y_test = self.y_rshp_[tr_val_smpl:, ...]
        
        if self.verbose == 1:
            print("Data split into train, val, and test.")

    def calibrate_model(self, run_model, msk_optim, val_nonoptim, StartParamDistribT, loss_func, maximize=False):
        """Calibre le modèle GR4J avec les données d'entraînement."""
        #beg_train = "1998-01-01"
        #end_train = "2008-12-14"
        self.dic_resultats["training_finished"] = False
        
        beg_train = np.min(self.date_train)   #quelles dates utiliser ?
        end_train = np.max(self.date_train)
        
        beg_test = np.min(self.date_test)
        end_test = np.max(self.date_test)
        
        self.dic_resultats["ti_train"] = beg_train
        self.dic_resultats["tf_train"] = end_train
        self.dic_resultats["ti_test"] = beg_test
        self.dic_resultats["tf_test"] = end_test

        self.nan_msk = np.where(np.isnan(self.data_cat.flow_mm.values), False, True)
        self.msk_train = (self.data_cat.index.values >= pd.to_datetime(beg_train)) * \
                    (self.data_cat.index.values <= pd.to_datetime(end_train)) * self.nan_msk
        self.msk_test = (self.data_cat.index.values >= pd.to_datetime(beg_test)) * \
                   (self.data_cat.index.values <= pd.to_datetime(end_test)) * self.nan_msk

        start_time = time.time()
        
        self.param_final, self.crit_final = calibration_PAPGR(
            run_model=run_model,
            msk_optim=msk_optim,
            val_nonoptim=val_nonoptim,
            StartParamDistribT=StartParamDistribT,
            data_cat=self.data_cat,
            msk_train=self.msk_train,
            loss_func=loss_func,
            maximize=maximize
        )
        self.temps_calibration = time.time() - start_time
        self.dic_resultats["param_final"] = self.param_final[0]
        self.dic_resultats["temps_calibration"] = self.temps_calibration
        self.dic_resultats["NSE_calibration"] = self.crit_final
        self.dic_resultats["training_finished"] = True
        
        if self.verbose == 1:
            print("--- Calibration completed in %s seconds ---" % str(round((self.temps_calibration), 4)))
        return self.param_final, self.crit_final
    
    
    def test_model(self, run_model):

        # GR4J
        sim_ = run_model(self.param_final, self.data_cat)
        preds_gr = sim_[self.msk_test]

        ## NSE in test + MAE (mm/d)
        # nse_test = NSE(self.obs_t, preds_gr)
        # mae_test = MAE(self.obs_t, preds_gr)
        # print("NSE in test:", nse_test)
        # print("MAE in test:", mae_test)
        print("NSE in test:", NSE(self.y_test, preds_gr))
        
        for i in range(50):
            print(f"obs: {self.y_test[i]} - pred: {preds_gr[i]}")
        
        self.dic_resultats['NSE_test'] = NSE(self.y_test, preds_gr)
        self.dic_resultats['MAE_test'] = MAE(self.y_test, preds_gr)

    
    def save_results(self):
        """Enregistre les résultats de la calibration dans un fichier."""
        
        #dossier
        dir_resultats = os.path.join(self.dir_proj, "resultats")
        if not os.path.exists(dir_resultats):
            os.makedirs(dir_resultats)
        
        # fichier
        result_file = os.path.join(dir_resultats, "resultats_GR4J.csv")
        nom_BV = self.file_name.split("_")[0]
        self.dic_resultats["BV"] = nom_BV
        
        colonnes_resultats = self.dic_resultats.keys()
        if not os.path.exists(result_file):
            with open(result_file, "w") as f:
                f.write(",".join(colonnes_resultats) + "\n")
        
        str_result = ",".join([str(self.dic_resultats[col]) for col in colonnes_resultats])
        with open(result_file, "a") as f:
            f.write(f"{str_result}\n")
        
        if self.verbose == 1:
            print(f"Results saved to {result_file}")



# test de la classe GR4JModel pour le dernier BV
print('file = ', ts_files[-1])
gr4j_model = GR4JModel(dir_proj=dir_proj, file_name=ts_files[-1])
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




































# ## choose one catchment
# file_cat = ts_files[-1]
# print(file_cat)

# ## read the data
# data_cat = pd.read_csv(f"{dir_data}/{file_cat}", sep = ";")
# print(data_cat.head())
# data_cat.columns = ['date', 'precipitation', 'evapotranspiration', 'flow_mm']

# ## convert in terms of datatype
# data_cat.index = pd.to_datetime(data_cat.date)

# ## choices for the input data
# input_feat = ["precipitation", "evapotranspiration"]
# output_feat = ["flow_mm"]






# ## sequence length (in days)
# seq_len = 30

# ## reshaping the data given the sequence length
# x_in  = np.array(data_cat.drop([*output_feat, "date"], axis = 1))
# y_out = np.array(data_cat[output_feat])
# print("shape of x_in:", x_in.shape)
# print("shape of y_out:", y_out.shape)

# ## reshaping
# x_rshp, y_rshp = reshape_data(x = x_in, y = y_out, seq_length = seq_len)
# date_rshp = data_cat.date[(seq_len-1):].values

# print("shape of reshaped input:", x_rshp.shape)
# print("shape of reshaped output:", y_rshp.shape)
# print("shape of the dates:", date_rshp.shape)







# ## removing the cases with nan values in the output
# ## for this case, we are SURE there are no nan values in the input --> to be adapted for general cases
# y_nan  = np.where(np.isnan(y_rshp), 1, 0)
# y_nan_ = np.sum(y_nan, axis= 1)

# date_rshp_, x_rshp_, y_rshp_ = date_rshp[y_nan_ == 0], x_rshp[y_nan_ == 0,...], y_rshp[y_nan_ == 0,...]
# print("shape of reshaped input--no nan values:", x_rshp_.shape)
# print("shape of reshaped output--no nan values:", y_rshp_.shape)
# print("shape of the dates--no nan values:", date_rshp_.shape)







# ## create the training, validation, and test datasets
# ## training, validation, and test proportions
# tr_p, val_p, test_p = 0.5, 0.2, 0.3
# n_samples = date_rshp_.size
# tr_smpl = int(tr_p*n_samples)
# tr_val_smpl = int((tr_p+val_p)*n_samples)

# ## split of the data
# date_train, x_train, y_train = date_rshp_[:tr_smpl], x_rshp_[:tr_smpl, ...], y_rshp_[:tr_smpl,...]
# date_val  , x_val  , y_val   = date_rshp_[tr_smpl:tr_val_smpl], x_rshp_[tr_smpl:tr_val_smpl, ...], y_rshp_[tr_smpl:tr_val_smpl,...]
# date_test , x_test , y_test  = date_rshp_[tr_val_smpl:], x_rshp_[tr_val_smpl:, ...], y_rshp_[tr_val_smpl:,...]

# print("shape of x_train and y_train:", x_train.shape, y_train.shape)
# print("shape of x_val and y_val    :", x_val.shape, y_val.shape)
# print("shape of x_test and y_test  :", x_test.shape, y_test.shape)







# ## date train and test
# beg_train = np.min(date_train)
# beg_train = "1998-01-01"
# end_train = np.max(date_train)
# end_train = "2008-12-14"
# beg_test = np.min(date_test)
# end_test = np.max(date_test)

# ## mask for training and test periods (removes the nan values)
# nan_msk   = np.where(np.isnan(data_cat.flow_mm.values), False, True)
# msk_train = (data_cat.index.values >= pd.to_datetime(beg_train))*(data_cat.index.values <= pd.to_datetime(end_train))*nan_msk
# msk_test  = (data_cat.index.values >= pd.to_datetime(beg_test))*(data_cat.index.values <= pd.to_datetime(end_test))*nan_msk
# print(sum(msk_train))
# print(sum(msk_test))

# print(data_cat.head())
# print('ok')

# ## calibration: which model, which parameters to calibrate, starting parameter distribution
# run_model    = run_GR4J
# msk_optim    = np.array([True, True, True, True])
# val_nonoptim = np.ones(4)*np.nan
# StartParamDistribT = np.array([[5.13, -1.60, 3.03, -9.05],
#                                 [5.51, -0.61, 3.74, -8.51],
#                                 [6.07, -0.02, 4.42, -8.06]])
# ## calibration: cost function and 
# loss_func_gr = RMSE
# maximize     = False

# ## running the calibration
# start_time = time.time()

# param_final, crit_final = calibration_PAPGR(run_model = run_model, 
#                                             msk_optim = msk_optim, 
#                                             val_nonoptim = val_nonoptim, 
#                                             StartParamDistribT = StartParamDistribT, 
#                                             data_cat = data_cat, 
#                                             msk_train = msk_train, 
#                                             loss_func = loss_func_gr, maximize = maximize)
# print("--- %s secondes ---" % str(round((time.time() - start_time),4)))