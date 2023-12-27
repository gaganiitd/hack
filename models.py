import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


def training(csv_file, stock_name):
    df = pd.read_csv("NIFTY.csv")
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
    torch.save(model, '/kaggle/working/{0}.pt'.format(stock_name))

def predictor(csv_file, model_path):
    model = torch.load(model_path)
    df = pd.read_csv(csv_file)
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
    test_x = torch.from_numpy(X[int(0.8 * X.shape[0]):]).float()
    days = 30

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