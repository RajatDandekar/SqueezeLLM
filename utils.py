import json

def jload(data_path):
    with open(data_path, 'r') as f:
        return json.load(f)
