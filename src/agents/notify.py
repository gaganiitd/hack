import os

import dotenv
from uagents import Agent, Context, Protocol
from uagents.setup import fund_agent_if_low

from messages.basic import Notification
from utils.email_utils import send_email

# Load environment variables
dotenv.load_dotenv()

# Create notify agent
NOTIFY_AGENT_SEED = os.getenv("NOTIFY_AGENT_SEED", "notify agent secret phrase")

notify_agent = Agent(
    name="notify",
    seed=NOTIFY_AGENT_SEED,
    port=8002,
    endpoint=["http://127.0.0.1:8002/submit"],
)

# Ensure the agent has enough funds
fund_agent_if_low(notify_agent.wallet.address())


# Function to generate the mail body from the template
def generate_context(msg: Notification):
    stock_name=msg.notif
    context = {
        "stock_name":stock_name
    }
    return context


# Create a protocol for notifications
notify_protocol = Protocol("Predict")


# Function to handle incoming notifications requests
@notify_protocol.on_message(model=Notification)
async def send_notification(ctx: Context, sender: str, msg: Notification):
    if msg.email == "default.email@gmail.com":
        ctx.logger.error(
            "No email provided. See README for instructions to setup email. Skipping notification."
        )
        return
    context = generate_context(msg)
    success, data = await send_email(msg.name, msg.email, context)
    if success:
        ctx.logger.info("Email sent successfully")
    else:
        ctx.logger.error(f"Error sending email: {data}")


# include protocol with the agent
notify_agent.include(notify_protocol)
