import pandas as pd
import time


async def train_model(inp_csv, stock):
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
