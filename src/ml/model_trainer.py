import pandas as pd
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

async def train_model(inp_csv, stock):
<<<<<<< HEAD
    try:
        print("processing", inp_csv)
        df = pd.read_csv(inp_csv)
        time.sleep(2)
        with open(f"models/{stock}.h5", "w") as f:
            f.write("model")
        print("model saved for", stock)
        return True, None
    except Exception as e:
        return False, e
=======
    print("processing", inp_csv)
    df = pd.read_csv(inp_csv)
    time.sleep(2)
    df.to_csv(f"csv/{stock}_preds.csv", index=False)
    print("done")


df = pd.read_csv("/kaggle/input/niftycsv/NIFTY.csv")
closed_prices = df["Close"]

seq_len = 180

mm = MinMaxScaler()
scaled_price = mm.fit_transform(np.array(closed_prices)[... , None]).squeeze()

X = []
y = []

for i in range(len(scaled_price) - seq_len):
    X.append(scaled_price[i : i + seq_len])
    y.append(scaled_price[i + seq_len])

X = np.array(X)[... , None]
y = np.array(y)[... , None]
    
train_x = torch.from_numpy(X[:int(0.8 * X.shape[0])]).float()
train_y = torch.from_numpy(y[:int(0.8 * X.shape[0])]).float()
test_x = torch.from_numpy(X[int(0.8 * X.shape[0]):]).float()
test_y = torch.from_numpy(y[int(0.8 * X.shape[0]):]).float()

class Model(nn.Module):
    def __init__(self , input_size , hidden_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size , hidden_size , batch_first = True)
        self.fc = nn.Linear(hidden_size , 1)
    def forward(self , x):
        output , (hidden , cell) = self.lstm(x)
        return self.fc(hidden[-1 , :])
model = Model(1 , 64)

optimizer = torch.optim.Adam(model.parameters() , lr = 0.001)
loss_fn = nn.MSELoss()

num_epochs = 100

for epoch in range(num_epochs):
    output = model(train_x)
    loss = loss_fn(output , train_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0 and epoch != 0:
        print(epoch , "epoch loss" , loss.detach().numpy())

model.eval()
with torch.no_grad():
    output = model(test_x)

pred = mm.inverse_transform(output.numpy())
real = mm.inverse_transform(test_y.numpy())
days = 60

future_pred = test_x[-1]
print(future_pred.shape)
lst = []

for i in range(days):
    with torch.no_grad():
        output = model(future_pred.unsqueeze(0))
        lst.append(mm.inverse_transform(output.numpy()))
#         print(future_pred)
        future_pred = torch.cat((future_pred,output),0)
        future_pred = future_pred[1:]

    
print(lst)
plt.plot(pred.squeeze() , color = "red" , label = "predicted")
plt.plot(real.squeeze() , color = "green" , label = "real")
plt.show()
>>>>>>> 01bc74305cbd0225effe437c3352b472135616c1
