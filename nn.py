# %%
import pandas as pd 
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import MinMaxScaler


dtc_data = pd.read_csv(r'Radar_Traffic_Counts.csv')
#%%
dtc_data['location_name'] = dtc_data \
    .location_name.apply(lambda x: x.strip()) 


dtc_data['Date']=pd.to_datetime(dtc_data[['Year','Month','Day']])#,'Hour']])
df = dtc_data.groupby(['Date']) \
    .agg({'Volume':'sum'}) \
    .reset_index()
d_hourly = df['Volume']
d_hourly.index = pd.to_datetime(df['Date'])
#d_hourly = d_hourly.diff().fillna(d_hourly[0]).astype(np.int64)

# %%
#hourly_vol = dtc_data.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_raw = dtc_data.groupby(['Hour']) \
    .agg({'Volume':'mean'}) \
    .reset_index()
hourly = hourly_raw['Volume'] 
hourly = hourly.diff().fillna(hourly[0]).astype(np.int64)

#%%
test_data_size = 5

train_data = hourly[:-test_data_size]
test_data = hourly[-test_data_size:]
scaler = MinMaxScaler()
scaler = scaler.fit(np.expand_dims(train_data, axis=1))
train_data = scaler.transform(np.expand_dims(train_data, axis=1))
test_data = scaler.transform(np.expand_dims(test_data, axis=1))


#%%
# normalising sequence data
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)

seq_length = 5
X_train, y_train = create_sequences(train_data, seq_length)
X_test, y_test = create_sequences(test_data, seq_length)

X_train = torch.from_numpy(X_train).float()
y_train = torch.from_numpy(y_train).float()

X_test = torch.from_numpy(X_test).float()
y_test = torch.from_numpy(y_test).float()

# %%
class TrafficPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):

        self.n_hidden = n_hidden
        self.seq_len = seq_len
        self.n_layers = n_layers

        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=0.1
        )
        self.linear = nn.Linear(in_features=n_hidden, out_features=1)


    def forward(self,sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred