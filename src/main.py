from uagents import Bureau

from agents.trainer import trainer
from agents.predictor import predictor
from agents.notify import notify_agent

if __name__ == "__main__":
    bureau = Bureau(endpoint="http://127.0.0.1:8000/submit", port=8000)
    print("Adding predictor to bureau", predictor.address)
    bureau.add(predictor)
    print("Adding notify to bureau", notify_agent.address)
    bureau.add(notify_agent)
    print("Adding trainer to bureau", trainer.address)
    bureau.add(trainer)
    bureau.run()
