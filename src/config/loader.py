import json
import os 

current_dir_name = os.path.dirname(__file__)
config_file_path = os.path.join(current_dir_name, 'config.json')

def load(path):
    with open(path,'r') as f: 
        return json.load(f)
    
config = load(config_file_path)