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

## ----------------- LSTM Functions ---------------
@njit ## to accelerate
def reshape_data(x, y, seq_length):
    """
    Reshape matrix data into sample shape for LSTM training.

    x: matrix containing time steps as rows and input features as columns
    y: matrix containing time steps as rows and output features as columns
    seq_length: lookback
    """
    num_samples, num_features = x.shape
    x_new = np.zeros((num_samples - seq_length + 1, seq_length, num_features), np.float64)
    y_new = np.zeros((num_samples - seq_length + 1, 1),np.float64)

    for i in range(0, x_new.shape[0]):
        x_new[i, :, :num_features] = x[i:i + seq_length, :]
        y_new[i, :] = y[i + seq_length - 1, 0]

    return x_new, y_new

## MyDataset: PyTorch class for easy loading of data
class MyDataset(Dataset):
    def __init__(self, x_in, y_out):
        self.x_data = torch.from_numpy(x_in.astype(np.float32))
        self.y_data = torch.from_numpy(y_out.astype(np.float32))
        self.len = x_in.shape[0]

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

## LSTM model class
class LSTM_model(nn.Module):
    """Implementation of an LSTM network with ONE single layer + a dense layer
    """
    
    ## we need the __init__() method for inheritance from the nn.Module class
    def __init__(self, hidden_size, num_features, dropout_rate, nblayers):
        """create the model with user-defined hyperparameters
        hidden_size : number of LSTM cells
        num_features: number of input features
        dropout_rate: rate for regularization
        nb_layers   : number of LSTM layers
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.num_features = num_features
        self.nblayers = nblayers

        ## create the LSTM
        self.lstm = nn.LSTM(input_size = self.num_features, hidden_size = self.hidden_size, 
                            num_layers = self.nblayers, bias = True, batch_first = True)
        
        ## add a dropout layer
        self.dropout = nn.Dropout(p = self.dropout_rate)
        
        ## linear layer at the head
        self.fc = nn.Linear(in_features = self.hidden_size, out_features = 1)
    
    ## we need the forward method to run a prediction
    def forward(self, x):
        """forward the data through the network.
        x: input (torch.Tensor)
        """
        output, (h_n, c_n) = self.lstm(x)

        # perform prediction only at the end of the input sequence
        pred = self.fc(self.dropout(h_n[-1,:,:]))
        return pred


## function for training
def train_epoch(model, optimizer, train_loader, loss_func, epoch, device):
    """train model for a single epoch.
    model          : a torch.nn.Module implementing the LSTM model
    optimizer      : one of PyTorchs optimizer classes.
    train_loader   : a PyTorch DataLoader, providing the trainings
                     data in mini batches.
    loss_func      : the loss function to minimize.
    epoch          : the current epoch (int) used for the progress bar
    """
    ## set model to train mode (important for dropout)
    model.train()
    ## request mini-batch of data from the loader
    for x_, y_ in train_loader:
        ## delete previously stored gradients from the model
        optimizer.zero_grad()
        ## push data to GPU (if available)
        x_, y_ = x_.to(device), y_.to(device)
        ## get model predictions
        y_sim = model(x_)
        ## calculate loss
        loss = loss_func(y_sim, y_)
        ## calculate gradients
        loss.backward()
        ## update the weights
        optimizer.step()
    return loss.item()

## function to evaluate the model
def eval_model(model, dataloader, device):
    """Evaluate the model.
    model     : a torch.nn.Module implementing the LSTM model
    dataloader: pytorch dataloader
    """
    ## set the model to evaluation mode (important for dropout)
    model.eval()
    obs = []
    preds = []
    
    ## no backpropagation
    with torch.no_grad():
        # request mini-batch of data from the loader
        for x_, y_ in dataloader:
            ## push data to GPU (if available)
            x_ = x_.to(device)
            ## get model predictions
            y_sim = model(x_)
            obs.append(y_)
            preds.append(y_sim)
    return torch.cat(obs), torch.cat(preds)




## an EarlyStopper class
class EarlyStopper:
    ''' This class stops the training based on the evolution of the validation loss'''
    def __init__(self, patience = 1, min_delta = 0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss + self.min_delta < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


## ----------------- GR Functions ---------------
## transformation of parameters for GR4J
def transfo_param(params_in, dir_ = "R_to_T"):
    ''' transform parameters for efficient sampling of the loss function'''
    params_out = params_in.copy()
    if dir_ == "T_to_R":
        params_out[:,0] = np.exp(params_in[:,0])
        params_out[:,1] = np.sinh(params_in[:,1])
        params_out[:,2] = np.exp(params_in[:,2])
        params_out[:,3] = 20. + 19.5*(params_in[:,3] - 9.99)/19.98
    
    if dir_ == "R_to_T":
        params_out[:,0] = np.log(params_in[:,0])
        params_out[:,1] = np.arcsinh(params_in[:,1])
        params_out[:,2] = np.log(params_in[:,2])
        params_out[:,3] = 9.99 + 19.98*(params_in[:,3] - 20.)/19.5
    return params_out

## function to propose new candidates of parameter sets from a distribution (Latin Hypercube)
def NewCandidates(distribParam):
    '''provides candidates of parameter from a distribution of parameter values'''
    #x1, x2, x3, x4 = np.meshgrid(params_in[:,0], params_in[:,1], params_in[:,2], params_in[:,3])
    #new_array = np.array([x1.flatten(), x2.flatten(), x3.flatten(), x4.flatten()])
    new_array = np.array([y.flatten() for y in np.meshgrid(*[x for x in distribParam.T])])
    return np.unique(new_array.T, axis = 0)

## function to propose new candidates of parameter sets in the vicinity of a parameter set
def NewCandidatesLocal(new_param, search_ranges, msk_optim, pace):
    '''local sampling in all directions in the vicinity of a parameter set'''
    if(new_param.shape[0] != 1):
        print("You should send only one parameter set to this function.\n") 
        sys.exit(1)
    nparam = new_param.shape[1]
    vector_params = np.ones((2*nparam,nparam))*np.nan
    i_r = 0
    for i in np.arange(nparam):
        #print(i)
        if msk_optim[i]:
            for sgn in [-1.,+1.]:
                potential_param = new_param.copy()
                potential_param[0, i] = new_param[0,i] + sgn*pace
                potential_param[0, :] = np.where(potential_param[0, :] < search_ranges[0, :], search_ranges[0, :], potential_param[0, :])
                potential_param[0, :] = np.where(potential_param[0, :] > search_ranges[1, :], search_ranges[1, :], potential_param[0, :])
                vector_params[i_r,:] = potential_param
                i_r += 1
    sum_param = np.sum(vector_params, axis = 1)
    vector_params_ = vector_params[np.isnan(sum_param) == False, :]
    return np.unique(vector_params_, axis = 0)

## function to optimize GR4J (~Calibration_Michel in airGR)
def calibration_PAPGR(run_model, msk_optim, val_nonoptim, StartParamDistribT, data_cat, 
                      msk_train, loss_func, maximize = True):
    ## parameters of the algorithm
    nparams = msk_optim.size
    pace = 0.64
    pacediag = np.zeros((1,nparams))
    CLG = 0.7**(1/nparams)
    Compt = 0
    nruns = 0
    HistParamR = np.ones((101*nparams,nparams))*np.nan
    HistParamT = HistParamR.copy()
    HistCrit   = np.ones((101*nparams,1))*np.nan
    critOpt    = -np.inf if maximize else np.inf
    multip     = 1 if maximize else -1

    ## parameter distributions
    StartParamDistribR = transfo_param(StartParamDistribT, "T_to_R")

    SearchRgT = np.array([np.ones(nparams)*-9.99, np.ones(nparams)*9.99])
    SearchRgR = transfo_param(SearchRgT, "T_to_R")

    ## opt out for non-optimized parameters
    StartParamDistribR[:, msk_optim == False] = np.inf
    CandidatParamR_list = NewCandidates(StartParamDistribR)
    for i_ in np.argwhere(msk_optim == False):
        CandidatParamR_list[:, i_] = val_nonoptim[i_]

    ## --- 1/2 grid search
    for iNew in np.arange(len(CandidatParamR_list)):
        Param_ = CandidatParamR_list[None, iNew,:]
        sim_ = run_model(Param_, data_cat)
        nruns += 1
        crit_ = loss_func(obs = data_cat.flow_mm.values[msk_train], sim = sim_[msk_train])
        if crit_*multip > critOpt*multip:
            iNewOpt = iNew
            critOpt = crit_
    print(f"End of global search | {nruns} runs, obj. function = {critOpt:.4f}")

    ## --- 2/2 starting the gradient descent
    new_param_R = CandidatParamR_list[None, iNewOpt,:]
    new_param_T = transfo_param(new_param_R, "R_to_T")
    old_param_T = new_param_T.copy()
    HistParamR[0,:] = new_param_R
    HistParamT[0,:] = new_param_T
    HistCrit[0,:] = critOpt
    iter_ = 1
    while iter_ < HistParamR.shape[0]:
        ## pace too small: exit the loop
        if pace < 0.01:
            break

        ## create new parameter sets by local descent
        CandidatParamT_list = NewCandidatesLocal(new_param_T, SearchRgT, msk_optim, pace)
        CandidatParamR_list = transfo_param(CandidatParamT_list, "T_to_R")
        Progress = False

        ## loop over the new parameter candidates in the vicinity of new_param
        for iNew in np.arange(CandidatParamR_list.shape[0]):
            Param_ = CandidatParamR_list[None, iNew,:]
            sim_ = run_model(Param_, data_cat)
            nruns += 1
            crit_ = loss_func(obs = data_cat.flow_mm.values[msk_train], sim = sim_[msk_train])
            if crit_*multip > critOpt*multip:
                Progress = True
                iNewOpt = iNew
                critOpt = crit_

        ## if there is progress: increase the pace
        if Progress:
            old_param_T = new_param_T.copy()
            new_param_T = CandidatParamT_list[None, iNewOpt, :]
            Compt += 1
            if Compt > 2*nparams:
                pace = pace * 2
                Compt = 0
            vect_pace = new_param_T - old_param_T
            pacediag = np.where(msk_optim == False, pacediag, CLG * pacediag + (1-CLG) * vect_pace)
        ## no progress: decrease the pace
        else:
            pace = pace/2
            Compt = 0

        ## diagonal screening
        if iter_ > 4*nparams:
            CandidatParamT = new_param_T + pacediag
            CandidatParamT = np.where(CandidatParamT < SearchRgT[0,:], SearchRgT[0,:], CandidatParamT)
            CandidatParamT = np.where(CandidatParamT > SearchRgT[1,:], SearchRgT[1,:], CandidatParamT)
            CandidatParamR = transfo_param(CandidatParamT, "T_to_R")
            #print(CandidatParamR.shape)
            Param_ = CandidatParamR[None, 0,:]
            sim_ = run_model(Param_, data_cat)
            nruns += 1
            crit_ = loss_func(obs = data_cat.flow_mm.values[msk_train], sim = sim_[msk_train])
            if crit_*multip > critOpt*multip:
                critOpt = crit_
                old_param_T = new_param_T.copy()
                new_param_T = CandidatParamT.copy()

        ## save the results of the screening
        new_param_R = transfo_param(new_param_T, "T_to_R")
        HistParamR[iter_,:] = new_param_R
        HistParamT[iter_,:] = new_param_T
        HistCrit[iter_,:]   = critOpt
        iter_ += 1

    paramFinalR = new_param_R
    paramFinalT = new_param_T
    crit_final   = critOpt
    print(f"End of local search | {iter_ - 1} iterations, {nruns} runs, obj. function = {crit_final:.4f}")
    return paramFinalR, crit_final

## run GR4J
def run_GR4J(parameters, data_cat):
    '''run the GR4J model'''
    if parameters.size != 4:
        print("There should be four parameters to run GR4J.\n") 
        sys.exit(1)
    parameters = {"X1": parameters[0,0], "X2": parameters[0,1], "X3": parameters[0,2], "X4": parameters[0,3]}
    model = ModelGr4j(parameters)
    model.set_parameters(parameters)
    outputs = model.run(data_cat)
    return outputs["flow"].values

## ----------- Loss functions: one under pytorch and the equivalent for GR4J

## for LSTM
class loss_NSE(nn.Module):
    def __init__(self):
        super().__init__()  
    def forward(self, output, target):
        obs = target
        preds = output
        sum_errors = torch.sum( (obs - preds) **2)
        obs_m = torch.mean(obs)
        sum_vars = torch.sum( (obs - obs_m)**2 )
        return 1.-sum_errors/sum_vars
    
class loss_MAE(nn.Module) :
    def __init__(self):
        super().__init__()  
    def forward(self, output, target):
        obs = target
        preds = output
        
        sum_errors = torch.sum( torch.abs((obs - preds)))
        
        return torch.mean(sum_errors)
    
    
## for GR4J
def NSE(obs, sim):
    obs_m = np.mean(obs)
    N = np.sum((obs - sim)**2)
    D = np.sum((obs - obs_m)**2)
    return 1. - N/D

## for GR4J
def RMSE(obs, sim):
    return np.sqrt(np.mean((obs - sim)**2))

def MAE(obs, preds):
    return np.mean(np.abs(obs-preds))