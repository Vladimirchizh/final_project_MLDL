# %%
import pandas as pd 
import torch.nn as nn
dtc_data = pd.read_csv(r'Radar_Traffic_Counts.csv')
#%%
dtc_data['location_name'] = dtc_data \
    .location_name.apply(lambda x: x.strip()) 
# %%
hourly_vol = dtc_data.groupby(['location_name','location_latitude','location_longitude','Direction','Hour']).agg({'Volume':'mean'}).reset_index()
hourly_vol
#%%

# normalising 
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data)-seq_length-1):
        x = data[i:(i+seq_length)]
        y = data[i+seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


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