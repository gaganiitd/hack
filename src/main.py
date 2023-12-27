from uagents import Bureau

from agents.trainer import trainer
from agents.predictor import predictor

if __name__ == "__main__":
    bureau = Bureau(endpoint="http://127.0.0.1:8000/submit", port=8000)
    print("Adding predictor to bureau", predictor.address)
    bureau.add(predictor)
    print("Adding trainer to bureau", trainer.address)
    bureau.add(trainer)
    bureau.run()
