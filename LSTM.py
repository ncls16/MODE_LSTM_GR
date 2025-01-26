## libraries
import matplotlib.pyplot as plt
#from numba import njit
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from typing import Tuple, List
import os, re, time, sys
from tqdm.notebook import tqdm
#from hydrogr import InputDataHandler, ModelGr4j
import spotpy
import time
from fonction import *




class LSTM():
    def __init__(self,dir_proj, dir_results, file_BV, nom, seq_len, loss_fonction, tr_p=0.5, val_p=0.2, test_p=0.3, verbose=0):
        """
        Initialise la classe pour un seul bassin versant.

        :param dir_data: Répertoire contenant les fichiers de données.
        :param file_BV: Nom du fichier pour le bassin versant.
        :param seq_len: Longueur de la séquence pour le modèle.
        :param tr_p: Proportion des données pour l'entraînement.
        :param val_p: Proportion des données pour la validation.
        :param test_p: Proportion des données pour le test.
        """
        # chemins
        self.dir_proj       = dir_proj
        self.dir_results    = dir_results
        self.dir_data       = os.path.join(dir_proj, "data")
        self.file_BV        = file_BV
        
        if not os.path.exists(self.dir_data):
            os.makedirs(self.dir_data)
        if not os.path.exists(self.dir_results):
            os.makedirs(self.dir_results)
        
        # paramètres
        self.seq_len        = seq_len
        self.tr_p           = tr_p
        self.val_p          = val_p
        self.test_p         = test_p
        self.verbose        = verbose
        self.loss_fonction  = loss_fonction
        
        # initialisation
        self.data_cat       = None
        self.date_rshp_     = None
        self.x_rshp_        = None
        self.y_rshp_        = None
        
        # lecture données
        self.input_feat = ["precipitation", "evapotranspiration"]
        self.output_feat = ["flow_mm"]
        
        # enregistrment résultats
        colonnes_resultats = ["BV", "seq_len", "ti_train", "tf_train", "ti_test", "tf_test", "NSE_train", "MAE_train", "NSE_val", "MAE_val", "NSE_test", "MAE_test", "training_finished","epoch","loss_fonction","training_time","nom"]
        self.dic_resultats = {col: None for col in colonnes_resultats}
        
        self.dic_resultats["nom"] = nom
    
    
    def load_data(self):
        """Charge les données pour le bassin versant spécifié."""
        file_path = f"{self.dir_data}/{self.file_BV}"
        self.data_cat = pd.read_csv(file_path, sep=";")
        self.data_cat.columns = ['date', 'precipitation', 'evapotranspiration', 'flow_mm']
        self.data_cat.index = pd.to_datetime(self.data_cat.date)
        
        if self.verbose == 1:
            print("Data loaded for basin:", self.file_BV)
    
    
    def preprocess_data(self):
        """Prétraite les données en créant les entrées et sorties pour le modèle."""
        self.dic_resultats["seq_len"] = self.seq_len
        
        x_in = np.array(self.data_cat.drop([*self.output_feat, "date"], axis=1))
        y_out = np.array(self.data_cat[self.output_feat])
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
        
        
        # enregistrement dates
        beg_train = np.min(self.date_train)
        end_train = np.max(self.date_train)
        
        beg_test = np.min(self.date_test)
        end_test = np.max(self.date_test)
        
        self.dic_resultats["ti_train"] = beg_train
        self.dic_resultats["tf_train"] = end_train
        self.dic_resultats["ti_test"] = beg_test
        self.dic_resultats["tf_test"] = end_test
        
    def standardization(self):
        ## standardization of the input data
        self.x_mean = np.array(self.data_cat[self.data_cat.date.isin(self.date_train)][self.input_feat].mean())
        self.x_std  = np.array(self.data_cat[self.data_cat.date.isin(self.date_train)][self.input_feat].std())
        self.x_mean_r = self.x_mean.reshape((1, self.x_mean.shape[0]))
        self.x_std_r  = self.x_std.reshape((1,self.x_std.shape[0]))

        self.x_train = (self.x_train - self.x_mean_r)/self.x_std_r
        self.x_val   = (self.x_val   - self.x_mean_r)/self.x_std_r
        self.x_test  = (self.x_test  - self.x_mean_r)/self.x_std_r
        
    def train(self, num_layers=1, hidden_size=128, batch_size=256, n_epochs=2000):
        start_training_time = time.time()
        ## device: cpu or cuda_gpu if availabled
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if self.verbose == 1:
            print('using device:', self.device)
        self.dic_resultats["training_finished"] = False

        ## hyperparameters for the model structure: 
        ##    number of layers, cells per layers, and number of input features
        num_feat    = len(self.input_feat)

        ## training hyperparameters: bacth size, max number of epochs, learning rate, optimizer and loss function
        lr         = 0.001
        dropout    = 0.4
        patience   = 100
        min_delta  = 0.0001
        
        
        ## create DataLoaders for efficient loading of data (proper to pytorch)
        ds_train = MyDataset(x_in = self.x_train, y_out = self.y_train)
        ds_val   = MyDataset(x_in = self.x_val,   y_out = self.y_val)
        ds_test  = MyDataset(x_in = self.x_test,  y_out = self.y_test)

        self.train_Loader = DataLoader(ds_train, batch_size = batch_size, shuffle = True) 
        self.val_Loader   = DataLoader(ds_val,   batch_size = batch_size, shuffle = False)
        self.test_Loader  = DataLoader(ds_test,  batch_size = batch_size, shuffle = False)
        
        
        ## model instantiation, loss function, and optimizer
        self.mymodel = LSTM_model(hidden_size = hidden_size, num_features = num_feat, 
                                        dropout_rate = dropout, nblayers = num_layers).to(self.device)
        
        #loss_func = nn.MSELoss()
        if self.loss_fonction == 'NSE':
            loss_func = loss_NSE()
            maximize = True
        
        elif self.loss_fonction == 'MAE':
            loss_func = loss_MAE()
            maximize = False
            
        self.dic_resultats['loss_fonction'] = self.loss_fonction
        
        
        optimizer = torch.optim.Adam(self.mymodel.parameters(), lr = lr, maximize = maximize)

        ## get an idea about the number of parameters in the model and their shapes
        (wi, wh, bi, bh) = self.mymodel.lstm.parameters()
        (wo, bo) = self.mymodel.fc.parameters()
        
        # print(wi.shape)
        # print(wh.shape)
        # print(bi.shape)
        # print(bh.shape)
        # print(wo.shape)
        # print(bo.shape)
        
        
        # -- Training --
        ## early stopper
        early_stopper = EarlyStopper(patience = patience, min_delta = min_delta)

        ## model training
        mypbar = tqdm(np.arange(n_epochs), position=0, leave=True, unit='epoch', ascii=True)
        val_mse_list = []
        start_time = time.time()
        
        for i_epoch in mypbar:
            # ------------------- training ---------------------
            train_loss   = train_epoch(self.mymodel, optimizer, self.train_Loader, loss_func, i_epoch+1, self.device)
            obs_, preds_ = eval_model(self.mymodel, self.train_Loader, self.device)
            obs_, preds_ = obs_.to(self.device), preds_.to(self.device)
            train_loss_  = loss_func(obs_, preds_).item()
            
            # scores
            obs_flat = obs_.cpu().numpy().flatten()
            preds_flat = preds_.cpu().numpy().flatten()
            MAE_train = MAE(obs_flat, preds_flat)
            NSE_train = NSE(obs_flat, preds_flat)
            self.dic_resultats['NSE_train'] = NSE_train
            self.dic_resultats['MAE_train'] = MAE_train
            
            
            # -------------------- validation -------------------
            obs_, preds_ = eval_model(self.mymodel, self.val_Loader, self.device)
            obs_flat = obs_.cpu().numpy().flatten()
            preds_flat = preds_.cpu().numpy().flatten()

            # scores
            val_mse      = np.mean((obs_flat - preds_flat)**2)
            val_mse_list.append(val_mse)    
                    
            MAE_val = MAE(obs_flat, preds_flat)
            NSE_val = NSE(obs_flat, preds_flat)
            self.dic_resultats['NSE_val'] = NSE_val
            self.dic_resultats['MAE_val'] = MAE_val
            
           
            mypbar.set_description(f'Training loss: {train_loss_:.3f} | Validation MSE: {val_mse:.4f} | Counter: {early_stopper.counter} | Progress')
            
            # -- early stopping -- 
            stop_ = early_stopper.early_stop(val_mse)
            if stop_:
                if self.verbose == 1:   
                    print('Done! Early stop at epoch', i_epoch)
                break
            # epoch d'entrainement
            self.dic_resultats['epoch'] = i_epoch
            
            
        if self.verbose == 1:
            print("--- %s minutes ---" % str(round((time.time() - start_time)/60,2)))
        
        end_training_time = time.time()
        self.dic_resultats["training_time"] = end_training_time - start_training_time
        # si l'entrainement s'est bien passé
        self.dic_resultats["training_finished"] = True
        
    
    
    def test_model(self):
        # LSTM
        obs_t, preds_lstm = eval_model(self.mymodel, self.test_Loader, self.device)
        obs_t, preds_lstm = obs_t.cpu().numpy().flatten(), preds_lstm.cpu().numpy().flatten()

        # # Save obs_t to a file
        # obs_file = os.path.join(dir_data, 'obs_t.csv')
        # np.savetxt(obs_file, obs_t, delimiter=",")


        ## NSE in test + MAE (mm/d)
        NSE_t_lstm = NSE(obs_t, preds_lstm)
        MAE_t_lstm = MAE(obs_t, preds_lstm)
        
        self.dic_resultats["NSE_test"] = NSE_t_lstm
        self.dic_resultats["MAE_test"] = MAE_t_lstm
        
        
    def save_results(self, name):
        """Enregistre les résultats de la calibration dans un fichier."""
        
        #dossier
        if not os.path.exists(self.dir_results):
            os.makedirs(self.dir_results)
        
        # fichier
        result_file = os.path.join(self.dir_results, f"resultats_LSTM_{name}.csv")
        nom_BV = self.file_BV.split("_")[0]
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
            
        # df = pd.read_csv(result_file, error_bad_lines=False)
        # return df
        
        
        
        

