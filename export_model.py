from experiment import VAEXperiment
import argparse
import yaml
from utils import get_model_module

import torch
import torch.backends.cudnn as cudnn
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

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

# model_module = get_model_module(exp_path, config)
model_module = get_model_module('', config)
model = model_module(**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=model, params=config)

# export model
print('Preparing to export model...')
script = experiment.to_torchscript()
torch.jit.save(script, f'{os.path.splitext(ckpt_path)[0]}.pth')
print('Model exported successfully!')
