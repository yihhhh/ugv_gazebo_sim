
import numpy as np
import os
import sys
import yaml
import csv
from easydict import EasyDict
import wandb

import torch


def wandb_init(project_name, group_id, exp_id):
    os.environ["WANDB_API_KEY"] = "1a2a4b937223367f6631a531c97f21d4f47b8356"
    os.environ["WANDB_SILENT"] = "true"
    wandb.init(project=project_name, entity='yihan', group=group_id, name=exp_id, allow_val_change=True)

def load_config(config_path="config.yml"):
    if os.path.isfile(config_path):
        f = open(config_path)
        dict = yaml.load(f, Loader=yaml.FullLoader)
        try:
            wandb.config.update(dict)
        except:
            print("wandb not initiated.")
        return EasyDict(dict)
    else:
        raise Exception("Configuration file is not found in the path: "+config_path)

def wandb_update(config):
    wandb.config.update(config, allow_val_change=True)