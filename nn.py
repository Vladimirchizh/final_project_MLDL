# %%
import torch
import pandas as pd
from torch import nn, optim
import matplotlib.pyplot as plt


import numpy as np
from sklearn.preprocessing import MinMaxScaler

# %%

dtc_data = pd.read_csv(r'Radar_Traffic_Counts.csv')
# %%
dtc_data['location_name'] = dtc_data \
    .location_name.apply(lambda x: x.strip())

dtc_data['Date'] = pd.to_datetime(dtc_data[['Year', 'Month', 'Day']])  # ,'Hour']])
df = dtc_data.groupby(['Date']) \
    .agg({'Volume': 'sum'}) \
    .reset_index()
d_hourly = df['Volume']
d_hourly.index = pd.to_datetime(df['Date'])
# d_hourly = d_hourly.diff().fillna(d_hourly[0]).astype(np.int64)

# %% mean traffic for the whole city

hourly_raw = dtc_data.groupby(['Hour']) \
    .agg({'Volume': 'mean'}) \
    .reset_index()
hourly = hourly_raw['Volume']
hourly = hourly.diff().fillna(hourly[0]).astype(np.int64)

# %% traffic for each crossroad

locations = list(dtc_data.location_name.unique())
hourly_locations = dict()
for i in range(len(locations)):
    hourly_locations[i] = dtc_data[dtc_data['location_name'] == locations[i]] \
        .groupby(['Hour']) \
        .agg({'Volume': 'mean'}) \
        .reset_index()['Volume']
    hourly_locations[i] = hourly_locations[i].diff().fillna(hourly_locations[i][0]).astype(np.int64)

# %%
def test_train_scaling(dataset):
    test_data_size = 8
    train_data = dataset[:-test_data_size]
    test_data = dataset[-test_data_size:]
    # normalizing data
    scaler = MinMaxScaler()
    scaler = scaler.fit(np.expand_dims(train_data, axis=1))
    train_data = scaler.transform(np.expand_dims(train_data, axis=1))
    test_data = scaler.transform(np.expand_dims(test_data, axis=1))
    return train_data, test_data, scaler



# %%
# normalising sequence data
def create_sequences(data, seq_length):
    xs = []
    ys = []

    for i in range(len(data) - seq_length - 1):
        x = data[i:(i + seq_length)]
        y = data[i + seq_length]
        xs.append(x)
        ys.append(y)

    return np.array(xs), np.array(ys)


seq_length = 5
def sampling_data(train_data, test_data):
    X_train, y_train = create_sequences(train_data, seq_length)
    X_test, y_test = create_sequences(test_data, seq_length)    
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.from_numpy(y_train).float()

    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    return X_train, y_train, X_test, y_test



# %%
class TrafficPredictor(nn.Module):
    def __init__(self, n_features, n_hidden, seq_len, n_layers):
        super(TrafficPredictor,self).__init__()
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

    def forward(self, sequences):
        lstm_out, self.hidden = self.lstm(
            sequences.view(len(sequences), self.seq_len, -1),
            self.hidden
        )
        last_time_step = lstm_out.view(self.seq_len, len(sequences), self.n_hidden)[-1]
        y_pred = self.linear(last_time_step)
        return y_pred
    def reset_hidden_state(self):
        self.hidden = (
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden),
        torch.zeros(self.n_layers, self.seq_len, self.n_hidden)
        )

def train_model(model,train_data,train_labels, test_data=None,test_labels=None):
    loss_fn = torch.nn.MSELoss(reduction='sum')

    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    num_epochs = 200

    train_hist = np.zeros(num_epochs)
    test_hist = np.zeros(num_epochs)

    for t in range(num_epochs):
        model.reset_hidden_state()

        y_pred = model(X_train)

        loss = loss_fn(y_pred.float(), y_train)

        if test_data is not None:
            with torch.no_grad():
                y_test_pred = model(X_test)
                test_loss = loss_fn(y_test_pred.float(), y_test)
            test_hist[t] = test_loss.item()

            if t % 10 == 0:
                print(f'Epoch {t} train loss: {loss.item()} test loss: {test_loss.item()}')
        elif t % 10 == 0:
            print(f'Epoch {t} train loss: {loss.item()}')

        train_hist[t] = loss.item()

        optimiser.zero_grad()

        loss.backward()

        optimiser.step()

    return model.eval(), train_hist, test_hist
# %% 
train_data, test_data, scaler = test_train_scaling(hourly)
X_train, y_train, X_test, y_test = sampling_data( train_data, test_data)
model = TrafficPredictor(
  n_features=1, 
  n_hidden=512, 
  seq_len=seq_length, 
  n_layers=2
)
model, train_hist, test_hist = train_model(
  model, 
  X_train, 
  y_train, 
  X_test, 
  y_test
)

plt.plot(train_hist, label="Training loss")
plt.plot(test_hist, label="Test loss")
plt.ylim((0, 5))
plt.legend()
# %% 

with torch.no_grad():
  test_seq = X_test[:1]
  preds = []
  for _ in range(len(X_test)):
    y_test_pred = model(test_seq)
    pred = torch.flatten(y_test_pred).item()
    preds.append(pred)
    new_seq = test_seq.numpy().flatten()
    new_seq = np.append(new_seq, [pred])
    new_seq = new_seq[1:]
    test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

true_cases = scaler.inverse_transform(
    np.expand_dims(y_test.flatten().numpy(), axis=0)
).flatten()

predicted_cases = scaler.inverse_transform(
  np.expand_dims(preds, axis=0)
).flatten()
# %%
plt.plot(
  hourly.index[:len(train_data)], 
  scaler.inverse_transform(train_data).flatten(),
  label='Historical traffic data'
)

plt.plot(
  hourly.index[len(train_data):len(train_data) + len(true_cases)], 
  true_cases,
  label='Real traffic data'
)

plt.plot(
  hourly.index[len(train_data):len(train_data) + len(true_cases)], 
  predicted_cases, 
  label='Predicted traffic data'
)

plt.legend()
# %%
true_cases = dict()
predicted_cases = dict()
for z in range(len(hourly_locations)):
    train_data, test_data, scaler = test_train_scaling(hourly_locations[z])
    X_train, y_train, X_test, y_test = sampling_data(train_data, test_data )
    model = TrafficPredictor(n_features=1, n_hidden=512, seq_len=seq_length, n_layers=2)
    model, train_hist, test_hist = train_model(model, X_train, y_train, X_test, y_test)
    
    with torch.no_grad():
      test_seq = X_test[:1]
      preds = []
      for _ in range(len(X_test)):
        y_test_pred = model(test_seq)
        pred = torch.flatten(y_test_pred).item()
        preds.append(pred)
        new_seq = test_seq.numpy().flatten()
        new_seq = np.append(new_seq, [pred])
        new_seq = new_seq[1:]
        test_seq = torch.as_tensor(new_seq).view(1, seq_length, 1).float()

    true_cases[z] = scaler.inverse_transform(np.expand_dims(y_test.flatten().numpy(), axis=0)).flatten()
    predicted_cases[z] = scaler.inverse_transform(np.expand_dims(preds, axis=0)).flatten()
    
# %%
