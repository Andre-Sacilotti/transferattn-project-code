import random
import os
import numpy as np
import torch
import yaml
from src.mmodel import get_model
from src.mmodel.basic_params import basic_parser
from src.mmodel.ZZZ_model.params import params
from argparse import Namespace




HIDDEN_SIZE = 512
LAYERS = 4
HEADS = 8
DROPOUT = 0.2
MODEL = "ZZZ_model"

torch.cuda.empty_cache()
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

if params.dataset == "UCF_HMDB_I3D":
    if params.source == "ucf":
        config_file = "config_uh_i3d.yml"
    else:
        config_file = "config_hu_i3d.yml"
elif params.dataset == "UCF_HMDB_STAM":
    if params.source == "ucf":
        config_file = "config_uh_stam.yml"
    else:
        config_file = "config_hu_stam.yml"
elif params.dataset == "K_NEC":
    config_file = "config_nec.yml"
os.environ["config_file"] = config_file

with open("src/mmodel/basic_config.yml") as f:
    basic_parser = yaml.safe_load(f)

with open(f"src/mmodel/ZZZ_model/{config_file}") as f:
    experiment_parser = yaml.safe_load(f)


params = vars(params)
params.update(basic_parser)
params.update(experiment_parser)
params = Namespace(**params)

if params.random_seed is not None:
    torch.manual_seed(params.random_seed)
    torch.cuda.manual_seed_all(params.random_seed)
    np.random.seed(params.random_seed)
    random.seed(params.random_seed)

model = get_model(
    hidden_size=HIDDEN_SIZE,
    num_layers=LAYERS,
    num_heads=HEADS,
    dropout=DROPOUT,
    params=params
)

model.train_model()
