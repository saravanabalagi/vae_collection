from posixpath import split
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from datasets import OxfordRobotcar
from torch.utils.data import DataLoader
import os


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.save_hyperparameters(params)
        self.curr_device = None
        self.hold_graph = False
        try:
            self.hold_graph = self.params['exp_params']['retain_first_backpass']
        except:
            pass

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['exp_params']['batch_size']/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)

        # enable to log every step
        # self.logger.experiment.log({key: val.item() for key, val in train_loss.items()})
        return train_loss

    def training_epoch_end(self, outputs):
        self.log_weights_histogram()
        mean_outputs = self.mean_metrics(outputs)
        self.log_dict(mean_outputs)

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        val_loss = self.model.loss_function(*results,
                                            M_N = self.params['exp_params']['batch_size']/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        return val_loss

    def validation_epoch_end(self, outputs):
        self.sample_images()
        mean_outputs = self.mean_metrics(outputs)
        mean_outputs['val_loss'] = mean_outputs['loss']
        del mean_outputs['loss']
        self.log_dict(mean_outputs)

    def mean_metrics(self, dict_list):
        metrics = {}
        for k in dict_list[0].keys():
            metrics[k] = torch.Tensor([e[k] for e in dict_list]).mean()
        return metrics
    
    def log_weights_histogram(self):
        # iterating through all parameters
        for name,params in self.named_parameters():
            self.logger.experiment.add_histogram(name,params, self.current_epoch)

    def sample_images(self):
        # Get sample reconstruction image
        test_input, test_label = next(iter(self.sample_dataloader))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)
        recons = self.model.generate(test_input, labels = test_label)

        logger = self.logger
        img_dir = os.path.join(logger.save_dir, logger.name, f'version_{logger.version}', 'media')

        vutils.save_image(test_input, 
                    os.path.join(img_dir, f"epoch_{self.current_epoch}_input.png"),
                    normalize=True,
                    nrow=12)

        vutils.save_image(recons.data,
                          os.path.join(img_dir, f"epoch_{self.current_epoch}_recons.png"),
                          normalize=True,
                          nrow=12)

        try:
            samples = self.model.sample(144,
                                        self.curr_device,
                                        labels = test_label)
            vutils.save_image(samples.cpu().data,
                              os.path.join(img_dir, f"epoch_{self.current_epoch}_samples.png"),
                              normalize=True,
                              nrow=12)
        except:
            pass

        del test_input, recons #, samples


    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['exp_params']['LR'],
                               weight_decay=self.params['exp_params']['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['exp_params']['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['exp_params']['submodel']).parameters(),
                                        lr=self.params['exp_params']['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['exp_params']['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['exp_params']['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['exp_params']['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['exp_params']['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims

    def train_dataloader(self):
        transform = self.data_transforms(split='train')
        if self.params['exp_params']['dataset'] == 'oxford_robotcar':
            dataset = OxfordRobotcar(root = self.params['exp_params']['data_path'],
                             split="train",
                             transform=transform,
                             download=False)
                            
        else:
            raise ValueError('Undefined dataset type')

        self.num_train_imgs = len(dataset)
        return DataLoader(dataset,
                          num_workers=self.params['exp_params']['num_workers_dataloader'],
                          batch_size= self.params['exp_params']['batch_size'],
                          shuffle = True,
                          drop_last=True)

    def val_dataloader(self):
        transform = self.data_transforms(split='valid')
        if self.params['exp_params']['dataset'] == 'oxford_robotcar':
            self.sample_dataloader =  DataLoader(OxfordRobotcar(root = self.params['exp_params']['data_path'],
                                                        split="valid",
                                                        transform=transform,
                                                        download=False),
                                                 num_workers=self.params['exp_params']['num_workers_dataloader'],
                                                 batch_size= 144,
                                                 shuffle = False,
                                                 drop_last=True)
            self.num_val_imgs = len(self.sample_dataloader)
        else:
            raise ValueError('Undefined dataset type')

        return self.sample_dataloader

    def data_transforms(self, split):

        SetRange = transforms.Lambda(lambda X: 2 * X - 1.)
        SetScale = transforms.Lambda(lambda X: X/X.sum(0).expand_as(X))

        if self.params['exp_params']['dataset'] == 'oxford_robotcar':

            transforms_split = []
            if split == 'train':
                transforms_split = [
                    # transforms.RandomHorizontalFlip(),
                    # transforms.RandomGrayscale(),
                    # transforms.CenterCrop(148),
                    # transforms.RandomCrop(self.params['exp_params']['img_size']),
                    # transforms.RandomResizedCrop(self.params['exp_params']['img_size'], scale=(0.3, 1.0), ratio=(0.75, 1.3333333333333333)),
                    transforms.Resize(self.params['exp_params']['img_size']),
                    transforms.ToTensor(),
                    SetRange
                ]
            else:
                transforms_split = [
                    transforms.Resize(self.params['exp_params']['img_size']),
                    transforms.ToTensor(),
                    SetRange
                ]
            transform = transforms.Compose(transforms_split)
        else:
            raise ValueError('Undefined dataset type')
        return transform

