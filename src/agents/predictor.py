import json

from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

from ml.model_predictor import predict_model
from messages.basic import PredictMessage, PredictStatusMessage
from utils.email_utils import send_email
predictor = Agent(
    name="predictor",
    seed="predictor secret phrase",
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)

# Function to generate the mail body from the template
def generate_context(stock):
    stock_name=stock
    context = {
        "stock_name":stock_name
    }
    return context



fund_agent_if_low(predictor.wallet.address())

predictor_protocol = Protocol("Predict")
# notify_protocol = Protocol("Notify")

@predictor_protocol.on_message(model=PredictMessage)
async def predict(ctx: Context, sender: str, msg: PredictMessage):
    # Predict
    status, error = await predict_model(msg.stock)
    await ctx.send(
        sender, PredictStatusMessage(stock=msg.stock, status=status, error=error)
    )
    if status:
        context = generate_context(msg)
        success, data = await send_email("User", "aveinahsnahg@gmail.com", context)
        if success:
            ctx.logger.info("Email sent successfully")
        else:
            ctx.logger.error(f"Error sending email: {data}")


predictor.include(predictor_protocol)
# predictor.include(notify_protocol)