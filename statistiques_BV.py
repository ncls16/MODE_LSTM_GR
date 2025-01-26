import pandas as pd
import numpy as np
import os
import re

def read_data(dir_data, file_name):
    ## read the data
    data = pd.read_csv(f"{dir_data}/{file_name}", sep = ";")
    data.columns = ['date', 'precipitation', 'evapotranspiration', 'flow_mm']
    ## convert in terms of datatype
    data.index = pd.to_datetime(data.date)

    ## choices for the input data
    input = ["precipitation", "evapotranspiration"]
    output = ["flow_mm"]
    return data, input, output

## getting the data repo and the list of files
dir_proj = re.sub("\\\\","/", os.getcwd())
dir_data = f"{dir_proj}/data"
dir_resultats = f"{dir_proj}/resultats"
ts_files = os.listdir(dir_data)
ts_files = [file for file in ts_files if file.endswith("tsdaily.csv")]
code = [file[0:8] for file in ts_files]

## create a statistic dataframe
df_stats = pd.DataFrame( index = range(len(ts_files)),
                         columns = [ "BV",
                                     "mean_precip", "min_precip", "max_precip",
                                     "mean_ETP", "min_ETP", "max_ETP",
                                     "mean_flow_mm", "min_flow_mm", "max_flow_mm"])

## choose one catchment
file_cat = ts_files[-1]

## loop over the catchments
for i, file_cat in enumerate(ts_files):

    ## read the data
    data_cat, input_feat, output_feat = read_data(dir_data, file_cat)
    
    df_stats.loc[i, 'BV'] = file_cat[0:8]
    df_stats.loc[i, 'mean_precip'] = data_cat['precipitation'].mean()
    df_stats.loc[i, 'min_precip'] = data_cat['precipitation'].min()
    df_stats.loc[i, 'max_precip'] = data_cat['precipitation'].max()

    df_stats.loc[i, 'mean_ETP'] = data_cat['evapotranspiration'].mean()
    df_stats.loc[i, 'min_ETP'] = data_cat['evapotranspiration'].min()
    df_stats.loc[i, 'max_ETP'] = data_cat['evapotranspiration'].max()

    df_stats.loc[i, 'mean_flow_mm'] = data_cat['flow_mm'].mean()
    df_stats.loc[i, 'min_flow_mm'] = data_cat['flow_mm'].min()
    df_stats.loc[i, 'max_flow_mm'] = data_cat['flow_mm'].max()

## save the statistics
df_stats.to_csv(dir_resultats + "/statistiques.csv", index=False)
print("Le fichier 'statistiques.csv' a été généré avec succès.")