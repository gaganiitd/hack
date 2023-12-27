from uagents import Bureau

from agents.trainer import trainer

if __name__ == "__main__":
    bureau = Bureau()
    bureau.add(trainer)
    bureau.run()
