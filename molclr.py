import os
import shutil
import sys
import torch
import wandb
import yaml
import numpy as np
from datetime import datetime

import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR

from utils.nt_xent import NTXentLoss

from dataset.dataset_contrastive import MoleculeDatasetWrapper

apex_support = False
try:
    sys.path.append('./apex')
    from apex import amp

    apex_support = True
except:
    print("Please install apex for mixed precision training from: https://github.com/NVIDIA/apex")
    apex_support = False


def _save_config_file(model_checkpoints_folder):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        shutil.copy('./config.yaml', os.path.join(model_checkpoints_folder, 'config.yaml'))


class MolCLR(object):
    def __init__(self, dataset, config, loss='TripletMarginLoss'):
        self.config = config
        self.device = self._get_device()
        
        dir_name = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = os.path.join('ckpt', dir_name)
        self.writer = SummaryWriter(log_dir=log_dir)

        self.dataset = dataset
        self.loss = loss
        self.TripletMarginWithDistanceLoss = torch.nn.TripletMarginWithDistanceLoss(margin=1.0, distance_function=lambda x, y: 1.0 - F.cosine_similarity(x, y))
        # self.nt_xent_criterion = NTXentLoss(self.device, config['batch_size'], **config['loss'])

    def _get_device(self):
        if torch.cuda.is_available() and self.config['gpu'] != 'cpu':
            device = self.config['gpu']
            torch.cuda.set_device(device)
        else:
            device = 'cpu'
        print("Running on:", device)

        return device

    # def _step(self, model, xis, xjs, n_iter):
    def _step(self, model, anchor, positive, negative, n_iter):
        # # get the representations and the projections
        # ris, zis = model(xis)  # [N,C]

        # # get the representations and the projections
        # rjs, zjs = model(xjs)  # [N,C]

        # # normalize projection feature vectors
        # zis = F.normalize(zis, dim=1)
        # zjs = F.normalize(zjs, dim=1)

        # loss = self.nt_xent_criterion(zis, zjs)
        # return loss

        anchor_embedding = model(anchor)
        positive_embedding = model(positive)
        negative_embedding = model(negative)

        if self.loss == 'TripletMarginLoss':
            loss = F.triplet_margin_loss(anchor_embedding, positive_embedding, negative_embedding)
        elif self.loss == 'TripletMarginWithDistanceLoss':
            loss = self.TripletMarginWithDistanceLoss(anchor_embedding, positive_embedding, negative_embedding)

        return loss

    def train(self):
        wandb.init(project="USPTO50_Contrastive_GCN", name="gcn_triplet_cosine_distance_no_finetune", config=self.config)
        train_loader, valid_loader = self.dataset.get_data_loaders()

        if self.config['model_type'] == 'gin':
            from models.ginet_molclr import GINet
            model = GINet(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        elif self.config['model_type'] == 'gcn':
            from models.gcn_molclr import GCN
            model = GCN(**self.config["model"]).to(self.device)
            model = self._load_pre_trained_weights(model)
        else:
            raise ValueError('Undefined GNN model.')
        print(model)
        
        optimizer = torch.optim.Adam(
            model.parameters(), self.config['init_lr'], 
            weight_decay=eval(self.config['weight_decay'])
        )
        scheduler = CosineAnnealingLR(
            optimizer, T_max=self.config['epochs']-self.config['warm_up'], 
            eta_min=0, last_epoch=-1
        )

        if apex_support and self.config['fp16_precision']:
            model, optimizer = amp.initialize(
                model, optimizer, opt_level='O2', keep_batchnorm_fp32=True
            )

        model_checkpoints_folder = os.path.join(self.writer.log_dir, 'checkpoints')

        # save config file
        _save_config_file(model_checkpoints_folder)

        best_valid_loss = np.inf

        for epoch_counter in range(self.config['epochs']):
            n_iter = 0
            valid_n_iter = 0
            # for bn, (xis, xjs) in enumerate(train_loader):
            for bn, (anchor, positive, negative) in enumerate(train_loader):
                optimizer.zero_grad()

                # xis = xis.to(self.device)
                # xjs = xjs.to(self.device)

                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # loss = self._step(model, xis, xjs, n_iter)
                loss = self._step(model, anchor, positive, negative, n_iter)

                if n_iter % self.config['log_every_n_steps'] == 0:
                    self.writer.add_scalar('train_loss', loss, global_step=n_iter)
                    self.writer.add_scalar('cosine_lr_decay', scheduler.get_last_lr()[0], global_step=n_iter)
                    # print(epoch_counter, bn, loss.item())
                    print(f"Epoch: {epoch_counter} | Batch: {bn} | Loss: {loss.item()}")

                wandb.log({"epoch": epoch_counter, "train_loss": loss.item()})

                if apex_support and self.config['fp16_precision']:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                optimizer.step()
                n_iter += 1

            # validate the model if requested
            if epoch_counter % self.config['eval_every_n_epochs'] == 0:
                valid_loss = self._validate(model, valid_loader)
                # print(epoch_counter, bn, valid_loss, '(validation)')
                print(f"VALIDATION | Epoch: {epoch_counter} | Batch: {bn} | Loss: {valid_loss}")
                wandb.log({"epoch": epoch_counter, "val_loss": valid_loss})
                if valid_loss < best_valid_loss:
                    # save the model weights
                    best_valid_loss = valid_loss
                    torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model.pth'))
            
                self.writer.add_scalar('validation_loss', valid_loss, global_step=valid_n_iter)
                valid_n_iter += 1
            
            if (epoch_counter+1) % self.config['save_every_n_epochs'] == 0:
                torch.save(model.state_dict(), os.path.join(model_checkpoints_folder, 'model_{}.pth'.format(str(epoch_counter))))

            # warmup for the first few epochs
            if epoch_counter >= self.config['warm_up']:
                scheduler.step()

    def _load_pre_trained_weights(self, model):
        try:
            # checkpoints_folder = os.path.join('./ckpt', self.config['load_model'], 'checkpoints')
            # state_dict = torch.load(os.path.join(checkpoints_folder, 'model.pth'))
            # model.load_state_dict(state_dict)
            # model.load_state_dict(torch.load('ckpt/pretrained_gcn/checkpoints/model.pth', map_location='cuda:0'))
            # print("Loaded pre-trained model with success.")
            print("Not loading any pre-trained model.")
        except FileNotFoundError:
            print("Pre-trained weights not found. Training from scratch.")

        return model

    def _validate(self, model, valid_loader):
        # validation steps
        with torch.no_grad():
            model.eval()

            valid_loss = 0.0
            counter = 0
            # for (xis, xjs) in valid_loader:
            for bn, (anchor, positive, negative) in enumerate(valid_loader):
                # xis = xis.to(self.device)
                # xjs = xjs.to(self.device)
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)

                # loss = self._step(model, xis, xjs, counter)
                loss = self._step(model, anchor, positive, negative, counter)
                valid_loss += loss.item()
                counter += 1
            valid_loss /= counter
        
        model.train()
        return valid_loss


def main():
    config = yaml.load(open("config.yaml", "r"), Loader=yaml.FullLoader)
    print(config)

    # if config['aug'] == 'node':
    #     from dataset.dataset import MoleculeDatasetWrapper
    # elif config['aug'] == 'subgraph':
    #     from dataset.dataset_subgraph import MoleculeDatasetWrapper
    # elif config['aug'] == 'mix':
    #     from dataset.dataset_mix import MoleculeDatasetWrapper
    # else:
    #     raise ValueError('Not defined molecule augmentation!')

    dataset = MoleculeDatasetWrapper(config['batch_size'], **config['dataset'])
    molclr = MolCLR(dataset, config)
    molclr.train()


if __name__ == "__main__":
    main()
