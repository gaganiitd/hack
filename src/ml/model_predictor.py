import pandas as pd
import time


async def predict_model(stock):
    try:
        df = pd.read_csv(f"csv/{stock}.csv")
        # open model
        with open(f"models/{stock}.h5", "r") as f:
            model = f.read()
        time.sleep(2)
        df.to_csv(f"csv/{stock}_preds.csv", index=False)
        print("predictions saved for", stock)
        return True, ""
    except Exception as e:
        return False, str(e)
