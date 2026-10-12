import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import yaml
import os

from .losses import *
from .models import *
from .models.backbones import *
from .models.backbones import get_backbone
from .loss_aggregator import LossAggregator
from .data import *


class AELoader():
    def __init__(self, config_path, model_only = False, load=True):
        
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        self.config = config

        general_configs = config.get("general")
        self.num_epochs = general_configs.get("num_epochs")
        self.validation_interval = general_configs.get("validation_interval")
        self.experiment_name = general_configs.get("experiment_name")
        self.model_name = general_configs.get("model")
        self.save_dir = general_configs.get("save_dir")
        self.res_dir = os.path.join(self.save_dir, self.experiment_name)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        total_steps=100000

        

        if not model_only:
            dataset_configs = config.get("dataset", {})
            if self.model_name == 'fm':
                self.dataset = TripleDataset(split = 'train', **dataset_configs)
                self.test_dataset = TripleDataset(split = 'test', **dataset_configs)
            elif self.model_name == 'condeae':
                self.dataset = CondeDataset(split = 'all', **dataset_configs)
                self.test_dataset = CondeDataset(split = 'test', **dataset_configs)
            elif self.model_name == 'residualcondeae':
                self.dataset = ResidualCondeDataset(split = 'all', **dataset_configs)
                self.test_dataset = ResidualCondeDataset(split = 'test', **dataset_configs)
            else:
                '''self.dataset = UnifiedDataset(split = 'all', **dataset_configs)
                self.test_dataset = UnifiedDataset(split = 'seap', **dataset_configs)'''
                self.dataset = WuFaultDataset(split = 'all', return_fault=False, **dataset_configs)
                self.test_dataset = WuFaultDataset(split = 'test', return_fault=False, **dataset_configs)
            
    
            dataloader_configs = config.get("dataloader")
            self.dataloader = DataLoader(self.dataset, shuffle=True, drop_last = True, **dataloader_configs)
            self.test_dataloader = DataLoader(self.test_dataset, shuffle=True, drop_last = True, **dataloader_configs)
            self.batch_size = dataloader_configs['batch_size']
            total_steps = self.num_epochs*len(self.dataloader)

        backbone_configs = config.get("backbone", {})
        backbone_name = backbone_configs.get("name")
        backbone_class = get_backbone(backbone_name)
        backbone_args = backbone_configs.get("args", {}).copy()
        functions = backbone_configs.get("functions", None)
        if functions is not None:
        
            # Inject resolved classes in one-liners with defaults
            backbone_args["activation"] = self.get_activation(functions.get("activation", "silu"))
            backbone_args["norm"] = self.get_norm(functions.get("norm", "batchorm2d"))
            backbone_args["encoder_final_activation"] = self.get_activation(functions.get("encoder_final_activation", "identity"))
            backbone_args["decoder_final_activation"] = self.get_activation(functions.get("decoder_final_activation", "identity"))
        
        
        self.backbone = backbone_class(**backbone_args)

        loss_configs = config.get("loss")
        losses_config = loss_configs.get("losses")
        self.aggregator = LossAggregator(losses_config, device = self.device)

        model_config = config.get("model")
        if self.model_name == 'autoencoder':
            self.model = Autoencoder(backbone = self.backbone, loss_aggregator = self.aggregator, device = self.device, save_dir = self.res_dir, total_steps=total_steps,  **model_config)
        elif self.model_name == 'condeae':
            self.model = CondeAE(backbone = self.backbone, loss_aggregator = self.aggregator, device = self.device, save_dir = self.res_dir, total_steps=total_steps,  **model_config)
        elif self.model_name == 'residualcondeae':
            self.model = ResidualCondeAE(backbone = self.backbone, loss_aggregator = self.aggregator, device = self.device, save_dir = self.res_dir, total_steps=total_steps,  **model_config)
        elif self.model_name == 'fm':
            emb_size=backbone_args.get("embed_dim")
            self.model = FM(backbone = self.backbone, loss_aggregator = self.aggregator, device = self.device, save_dir = self.res_dir, total_steps=total_steps, emb_size=emb_size, **model_config)
        
        ckpt_path = f'{self.res_dir}/{self.experiment_name}_best.pt'
        if os.path.exists(ckpt_path) and load:
            print("Loading Checkpoint")
            CKPT = torch.load(ckpt_path)
            new_CKPT = {}
            for key, value in CKPT.items():
                if self.model_name=='autoencoder':
                    if 'model' not in key:
                        key = 'model.' + key
                new_CKPT[key] = value
            self.model.load_state_dict(new_CKPT)

    def get_activation(self, name: str):
        name = name.lower()
        activations = {
            "relu": nn.ReLU,
            "leakyrelu": nn.LeakyReLU,
            "silu": nn.SiLU,
            "gelu": nn.GELU,
            "elu": nn.ELU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "identity": nn.Identity,
            "prelu": nn.PReLU,
            "softplus": nn.Softplus,
            "selu": nn.SELU,
            "hardtanh": nn.Hardtanh,
            "softsign": nn.Softsign,
            "logsigmoid": nn.LogSigmoid,
        }
        return activations.get(name)
    
    
    def get_norm(self, name: str):
        name = name.lower()
        norms = {
            "batchnorm2d": nn.BatchNorm2d,
            "instance2d": nn.InstanceNorm2d,
            "layernorm": nn.LayerNorm,
            "groupnorm": GNWrapper,
            "identity": nn.Identity,
        }
        return norms.get(name)
