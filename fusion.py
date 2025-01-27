import pandas as pd 
import os 


df_catchments = pd.read_csv('resultats/catchments_contours.csv')
df_results = pd.read_csv('resultats/resultats_merged.csv')

df_catchments.rename(columns = {'CODE' : 'BV_LSTM'}, inplace=True)


df_fusion = pd.merge(df_results, df_catchments, how ='left', on='BV_LSTM')

df_fusion.to_csv(os.path.join ('resultats/fusion_shp_resultats.csv'), index = False)
print("Fusion terminée")