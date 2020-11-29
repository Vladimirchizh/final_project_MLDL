# %%
import pandas as pd 
dtc_data = pd.read_csv(r'Radar_Traffic_Counts.csv')
#%%
dtc_data['location_name'] = dtc_data \
    .location_name.apply(lambda x: x.strip()) 
# %%
hourly_vol = dtc_data.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_vol
# %% 
