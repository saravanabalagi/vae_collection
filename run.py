import yaml
import argparse
import numpy as np
from models import *
from experiment import VAEXperiment
import torch.backends.cudnn as cudnn
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TestTubeLogger
from utils import OnCheckpointSaveConfigCode


parser = argparse.ArgumentParser(description='Generic runner for VAE models')
parser.add_argument('--config',  '-c',
                    dest="filename",
                    metavar='FILE',
                    help =  'path to the config file',
                    default='configs/vae.yaml')

args = parser.parse_args()
with open(args.filename, 'r') as file:
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

model_module = vae_models[config['model_params']['name']]
model = model_module(**config['model_params'])
experiment = VAEXperiment(model, config)

runner = Trainer(default_root_dir=f"{tt_logger.save_dir}",
                 min_epochs=1,
                 logger=tt_logger,
                 log_every_n_steps=50,
                 limit_train_batches=1.,
                 val_check_interval=1.,
                 num_sanity_val_steps=5,
                 callbacks=[OnCheckpointSaveConfigCode(model_module)],
                 **config['trainer_params'])

print(f"======= Training {config['model_params']['name']} =======")
runner.fit(experiment)
