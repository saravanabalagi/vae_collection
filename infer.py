from experiment import VAEXperiment
import argparse
import yaml
from utils import get_model_module, split_batches
from tqdm import tqdm

import torch
import torch.backends.cudnn as cudnn
import pytorch_lightning as pl
from pytorch_lightning.loggers import TestTubeLogger
import numpy as np
import os
from typing import Iterable
from PIL import Image


parser = argparse.ArgumentParser()
parser.add_argument('ckpt_path', help='path to checkpoint file')
parser.add_argument('img_dir', help='path to dir containing imgs')
args = parser.parse_args()

ckpt_path = args.ckpt_path
img_dir = args.img_dir
exp_path = os.path.dirname(os.path.dirname(ckpt_path))

# load config
config_file_path = os.path.join(exp_path, 'config.yaml')
with open(config_file_path, 'r') as file:
    try:
        config = yaml.safe_load(file)
    except yaml.YAMLError as exc:
        print(exc)

# For reproducibility
torch.manual_seed(config['logging_params']['manual_seed'])
np.random.seed(config['logging_params']['manual_seed'])
cudnn.deterministic = True
cudnn.benchmark = False

model_module = get_model_module(exp_path, config)
model = model_module(**config['model_params'])
experiment = VAEXperiment.load_from_checkpoint(ckpt_path, vae_model=model, params=config)

# prediction mode
model = model.cuda(device=0)
model.eval()
torch.set_grad_enabled(False)
experiment.freeze()

img_paths = os.listdir(img_dir)
img_extensions = ['.jpg', '.png']
img_paths = [i for i in img_paths if os.path.splitext(i)[-1] in img_extensions]
img_paths_batches = list(split_batches(img_paths, 32))
embs = {}

print(f'Found {len(img_paths)} images')

for batch_no, img_paths_batch in enumerate(tqdm(img_paths_batches, desc="Inference")):

    # prepare imgs
    imgs = []
    for img_path in img_paths_batch:
        img_path_full = os.path.join(img_dir, img_path)
        img = Image.open(img_path_full)
        transform = experiment.data_transforms()
        img_transformed = transform(img)
        imgs.append(img_transformed)

    # inference
    imgs_torch = torch.stack(imgs).cuda(device=0)
    result_batch = experiment.forward(imgs_torch)
    if isinstance(result_batch, Iterable) and len(result_batch) >= 3:
        emb_batch = result_batch[2]
    else: emb_batch = result_batch
    
    # store emb in dict
    for img_path, emb in zip(img_paths_batch, emb_batch):
        key = os.path.splitext(img_path)[0]
        emb_cpu = emb.cpu().numpy()
        embs[key] = np.array(emb_cpu, dtype=np.float32)

# Save as npz
npz_dir = os.path.join(exp_path, 'feats')
os.makedirs(npz_dir, exist_ok=True)
npz_path = os.path.join(npz_dir, f"{config['model_params']['name']}_{os.path.basename(img_dir)}")
np.savez_compressed(npz_path, **embs)
print(f'Saved feats at {npz_path}.npz')
