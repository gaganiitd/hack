import json
from collections import deque
from datetime import datetime
from dateutil.relativedelta import relativedelta

from yahoo_fin.stock_info import get_data
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

from ml.model_trainer import train_model
from messages.basic import PredictMessage, PredictStatusMessage


trainer = Agent(
    name="trainer",
    seed="trainer secret phrase",
    port=8001,
    endpoint=["http://127.0.0.1:8001/submit"],
)

fund_agent_if_low(trainer.wallet.address())

PREDICTOR_AGENT_ADDRESS = (
    "agent1q256larkpyqgcgqdrh424ng7kkvqynrxa0nrsyjwajny3xvpcmylcdx9kew"
)

predictor_protocol = Protocol("Predict")


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


def update_train_queue(ctx: Context):
    # Get train queue
    with open("json/train_queue.json", "r") as f:
        queue = json.load(f)
    if not queue:
        return

    # print(queue)
    # empty the queue
    # with open("json/train_queue.json", "w") as f:
    # json.dump([], f)

    # add to agent storage
    ctx.storage.queue.extend(queue)


@trainer.on_event("startup")
async def initialize_storage(ctx: Context):
    # initialize an empty deque on agent storage
    ctx.storage.queue = deque()


@predictor_protocol.on_interval(5, messages=PredictMessage)
async def train(ctx: Context):
    update_train_queue(ctx)
    # print("q", ctx.storage.queue)
    if ctx.storage.queue:
        stock = ctx.storage.queue.popleft()
        await get_csv(stock)
        # Train model
        status = train_model(f"csv/{stock}.csv", stock)
        # add to trained models.json
        if not status:
            ctx.logger.error(f"Failed to train for {stock}")
            return
        with open("json/trained_models.json", "r") as f:
            trained_models = json.load(f)
        trained_models.append(stock)
        with open("json/trained_models.json", "w") as f:
            json.dump(trained_models, f)
        ctx.logger.info(f"Trained for {stock}")
        # predict
        await ctx.send(
            PREDICTOR_AGENT_ADDRESS,
            PredictMessage(stock=stock),
        )


@predictor_protocol.on_message(model=PredictStatusMessage)
async def handle_prediction(ctx: Context, sender: str, msg: PredictStatusMessage):
    if msg.status:
        try:
            with open("json/predicted_stocks.json", "r") as f:
                predicted_stocks = json.load(f)
        except FileNotFoundError:
            predicted_stocks = []
        predicted_stocks.append(msg.stock)
        with open("json/predicted_stocks.json", "w") as f:
            json.dump(predicted_stocks, f)
        ctx.logger.info(f"Predictions for {msg.stock} done")
    else:
        ctx.logger.error(f"Failed to predict for {msg.stock}\nError: {msg.error}")


trainer.include(predictor_protocol)
