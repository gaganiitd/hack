import pandas as pd
def calculate_ema(data, column, n):
    ema_column = f'{n}ema'
    data[ema_column] = data[column].ewm(span=n, adjust=False).mean()
    return data


def calculate_macd(data, short_window, long_window, signal_window):
    data['Short_EMA'] = data['Close'].ewm(span=short_window, adjust=False).mean()
    data['Long_EMA'] = data['Close'].ewm(span=long_window, adjust=False).mean()
    data['MACD'] = data['Short_EMA'] - data['Long_EMA']
    data['Signal_Line'] = data['MACD'].ewm(span=signal_window, adjust=False).mean()
    return data

file_path = 'filename'
stock_data = pd.read_csv(file_path)
stock_data['Date'] = pd.to_datetime(stock_data['Date'])
stock_data.set_index('Date', inplace=True)

short_window = 12
long_window = 26
signal_window = 9

stock_data = calculate_ema(stock_data, 'Close', 50)
stock_data = calculate_ema(stock_data, 'Close', 100)
stock_data = calculate_macd(stock_data, short_window, long_window, signal_window)


