from pytorch_lightning.callbacks import Callback
from pytorch_lightning.core.saving import save_hparams_to_yaml
import inspect
import sys
import os
from shutil import copyfile
from models import vae_models
import subprocess


def get_model_module(exp_path, config):
    try:
        # load model from file
        model_module_path = os.path.join(exp_path, 'code', 'model.py')
        if not os.path.exists(model_module_path):
            raise IOError(f'{model_module_path} not found, using from current codebase')
        sys.path.append(os.path.join(exp_path, 'code'))
        import model
        model_module = model.__dict__[config['model_params']['name']]
    except Exception as e:
        print(f'Error: {e}')
        print(f'Could not load model file from {model_module_path}')
        print(f'Loading model using the current codebase\n')
        model_module = vae_models[config['model_params']['name']]
    return model_module


class OnCheckpointSaveConfigCode(Callback):
    def __init__(self, model_module) -> None:
        self.model_module = model_module
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        # only do this 1 time
        if trainer.current_epoch == 0:
            logger = trainer.logger
            exp_path = os.path.join(logger.save_dir, logger.name, f'version_{logger.version}')
            code_path = os.path.join(exp_path, 'code')
            os.makedirs(code_path, exist_ok=True)

            # save config file
            config_file = os.path.join(exp_path, 'config.yaml')
            save_hparams_to_yaml(config_yaml=config_file, hparams=pl_module.params)

            # copy model file
            model_file = inspect.getsourcefile(self.model_module)
            copyfile(model_file, os.path.join(code_path, 'model.py'))

            # copy code archive for reproducibility
            cmd = f'git ls-files | grep -v "^assets/" | tar -T - -czf {code_path}/code_archive.tar.gz'
            subprocess.call(cmd, shell=True)
