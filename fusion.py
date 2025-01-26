import pandas as pd 
import os 


df_catchments = pd.read_csv('catchments_contours.csv')
df_results = pd.read_csv('MODE_LSTM_GR/resultats/resultats_merged.csv')

df_catchments.rename(columns = {'CODE' : 'BV_LSTM'}, inplace=True)

#print(df_catchments.head)

df_fusion = pd.merge(df_results, df_catchments, how ='left', on='BV_LSTM')

df_fusion.to_csv(os.path.join ('/resultats/fusion_shp_resultats.csv'), index = False)
