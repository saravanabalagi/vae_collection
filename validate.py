from experiment import VAEXperiment
import argparse
import yaml
from utils import get_model_module

import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
import numpy as np
import os


parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path', help='path to checkpoint file')
args = parser.parse_args()

ckpt_path = args.ckpt_path
exp_path = os.path.dirname(os.path.dirname(ckpt_path))

# load config
with open(os.path.join(exp_path, 'config.yaml'), 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

tt_logger = TestTubeLogger(
    save_dir=config['logging_params']['save_dir'],
    name=config['logging_params']['name'],
    debug=False,
    create_git_tag=False,
)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model_module = get_model_module(exp_path, config)
model = model_module(**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=model, params=config)

trainer = pl.Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 logger=tt_logger,
                 **config['trainer_params'])
trainer.validate(experiment)
