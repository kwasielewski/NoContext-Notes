from functools import cached_property
import os
from typing import Dict
import torch
import random
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from mingpt.utils import set_seed
from mingpt.model import GPT
from mingpt.trainer import Trainer
from datetime import datetime

def init_model():
    model_config = GPT.get_default_config()
    model_config.model_type = 'gpt-nano'

    model_config.vocab_size = train_dataset.get_vocab_size()
    model_config.block_size = train_dataset.get_block_size()
    model = GPT(model_config)
    print (model_config.n_head, model_config.n_layer, model_config.n_embd)

    train_config = Trainer.get_default_config()
    train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
    train_config.max_iters = 10000
    train_config.num_workers = 0
    trainer = Trainer(train_config, model, train_dataset)

    return model, trainer

def main():
    torch.load()
