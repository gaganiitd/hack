from uagents import Model

# from typing import List, Dict, Tuple


class PredictMessage(Model):
    stock: str


class PredictStatusMessage(Model):
    stock: str
    status: bool
    error: str


class Notification(Model):
    
    
    name: str
    email: str
    notif: str
    
