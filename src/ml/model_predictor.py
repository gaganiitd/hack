import pandas as pd
import time


async def predict_model(stock):
    try:
        df = pd.read_csv(f"csv/{stock}.csv")
        # open model
        print("predictions saved for", stock)
        return True, ""
    except Exception as e:
        return False, str(e)
