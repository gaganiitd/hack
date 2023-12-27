import json

from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

from ml.model_predictor import predict_model
from messages.basic import PredictMessage, PredictStatusMessage

predictor = Agent(
    name="predictor",
    seed="predictor secret phrase",
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)

fund_agent_if_low(predictor.wallet.address())

predictor_protocol = Protocol("Predict")


@predictor_protocol.on_message(model=PredictMessage)
async def predict(ctx: Context, sender: str, msg: PredictMessage):
    # Predict
    status, error = await predict_model(msg.stock)
    await ctx.send(
        sender, PredictStatusMessage(stock=msg.stock, status=status, error=error)
    )


predictor.include(predictor_protocol)
