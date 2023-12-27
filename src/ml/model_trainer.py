import pandas as pd
import time
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from date_function import dates

# async def train_model(inp_csv, stock):
#     print("processing", inp_csv)
#     df = pd.read_csv(inp_csv)
#     time.sleep(2)
#     df.to_csv(f"csv/{stock}_preds.csv", index=False)
#     print("done")

def train_model(csv_file,stock, num_days=360):
    return True
    df = pd.read_csv(csv_file)
    closed_prices = df["close"]

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

    num_epochs = 20

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
    days = num_days

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
    kst, d = dates(num_days)
    dct = {}
    for i in range(num_days):
        dct[kst[i]] = lst[i][0][0]
    print(dct)
    # df = pd.read_csv(csv_file)
    print(df.tail())
    # df2 = pd.DataFrame()
    df2 = pd.DataFrame({"date":dct.keys(),"close":dct.values()})
    df2['ticker'] = stock
    df2.to_csv(f"./csv/{stock}_preds.csv",index=False)
    # r.to_csv(f"{stock}_preds.csv")
    plt.show()
    return True

# model_trainer('./../../HDFCBANK.NS.csv',"HDFCBANK.NS",20)
