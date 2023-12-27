import pandas as pd
import time


async def train_model(inp_csv, stock):
    print("processing", inp_csv)
    df = pd.read_csv(inp_csv)
    time.sleep(2)
    df.to_csv(f"csv/{stock}_preds.csv")
    print("done")
