import json
from datetime import datetime
from dateutil.relativedelta import relativedelta

from yahoo_fin.stock_info import get_data
from uagents import Agent, Context
from uagents.setup import fund_agent_if_low

from ml.model_trainer import train_model

trainer = Agent(
    name="trainer",
    seed="trainer secret phrase",
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)

fund_agent_if_low(trainer.wallet.address())


async def get_csv(stock):
    # Get csv data from API
    end_date = datetime.now()
    start_date = end_date - relativedelta(years=5)
    df = get_data(
        stock,
        start_date=start_date.strftime("%m/%d/%Y"),
        end_date=end_date.strftime("%m/%d/%Y"),
        index_as_date=False,
    )

    df = df.loc[:, ["date", "close", "ticker"]]
    df.to_csv(f"csv/{stock}.csv", index=False)


def get_train_queue():
    # Get train queue
    with open("json/train_queue.json", "r") as f:
        queue = json.load(f)
    print(queue)
    # empty the queue
    with open("json/train_queue.json", "w") as f:
        json.dump([], f)
    return queue


@trainer.on_interval(60)
async def train(ctx: Context):
    queue = get_train_queue()
    print("q", queue)
    for stock in queue:
        await get_csv(stock)
        # Train model
        await train_model(f"csv/{stock}.csv", stock)
        # add to trained models.json
        with open("json/trained_models.json", "r") as f:
            trained_models = json.load(f)
        trained_models.append(stock)
        with open("json/trained_models.json", "w") as f:
            json.dump(trained_models, f)
