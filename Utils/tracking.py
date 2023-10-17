import wandb
from typing import Dict

def init_tracker(name:str, config:Dict):
    wandb.init(project=name, 
               config=config)

def tracker_log(dct:Dict):
    wandb.log({**dct})

def tracker_summary(dct:Dict):
    for key in dct.keys():
        wandb.run.summary[key] = dct[key]