import json
import os

def load_config():
    with open("config.json", "r") as f:
        return json.load(f)

config = load_config()