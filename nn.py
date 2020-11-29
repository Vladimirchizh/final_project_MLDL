# %%
import pandas as pd 
dtc_data = pd.read_csv(r'Radar_Traffic_Counts.csv')
#%%
dtc_data['location_name'] = dtc_data \
    .location_name.apply(lambda x: x.strip()) 
 