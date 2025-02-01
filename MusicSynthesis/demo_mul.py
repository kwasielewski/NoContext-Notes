#!/usr/bin/env python
# coding: utf-8

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

set_seed(3407)
print("Loading trainer")

class AbcTokenizer:
    abc_julia_arr = [
        'Song',
        'Bar',
    ] + \
    [f'Note {i}' for i in range(-12, 13)] + \
    [f'Len {i}' for i in [1, 2, 4, 8]] + \
    [f'Len /{i}' for i in [1, 2, 4, 8]]

    @cached_property
    def abc_to_token(self) -> Dict[str, int]:
        return {tok:idx for idx, tok in enumerate(self.abc_julia_arr)}

    @cached_property
    def token_to_abc(self) -> Dict[int, str]:
        return {idx:tok for idx, tok in enumerate(self.abc_julia_arr)}

    def get_vocab(self) -> int:
        return len(self.abc_julia_arr)
    
    def encode(self, text: str) -> list[int]:
        return [self.abc_to_token[l] for l in text.split("\n")[:-1]]

    def decode(self, tok: int) -> str:
        return self.token_to_abc[tok]

class AbcDataset(Dataset):
    """ 
    Dataset for the Mult problem. E.g. for problem length 3:
    12 * 333 = 3996
    Input: 0 1 2 3 3 3 -> Output: 0 0 3 9 9 6
    Which will feed into the transformer concatenated as:
    input:  0 1 2 3 3 3 0 3 4
    output: I I I 0 0 3 9 9 6
    where I is "ignore", as the transformer is reading the input sequence
    """

    def __init__(self, split: str, dataset: str):
        assert split in {'train', 'test'}
        self.split = split
        self.datafiles = [d for d in os.listdir(dataset) if d.endswith(".abc")]
        self.datafiles_len = len(self.datafiles)
        self.sizes = {
            'train': self.datafiles_len - (self.datafiles_len // 4), 
            'test': self.datafiles_len
        }

        self.tokenizer = AbcTokenizer()

        random.shuffle(self.datafiles)
    
    def __len__(self):
        return self.sizes[self.split]
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab()
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return 10

    def load_file(self, filename: str):
        with open(filename, "r") as f:
            return self.tokenizer.encode(f.read())

    def get_inp(self, idx: int):
        match self.split:
            case 'train':
                return idx + (idx // 3)
            case 'test':
                return (idx + 1) * 4 - 1

    # 0 0 0
    # 1 1 0
    # 2 2 0
    # 3 4 1
    # 4 5 1
    # 5 6 1
    # 6 8 2
    # 7 9 2
    # 8 10 2
    # 9 12 3
    # 10 13 3

    # 0 3
    # 1 7
    # 2 11
    # 3 15

    def __getitem__(self, idx):
        rai = self.load_file(self.datafiles[get_inp]) 
        h = hash(str(rai[:10]))
            
        inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
        #if inp_split == self.split:
        #    break # ok
        
        x = torch.tensor(rai[:-1], dtype=torch.long)
        y = torch.tensor(rai[1:], dtype=torch.long)
        
        # we only want to predict at output locations, mask out the loss at the input locations
        y[:2*self.length - 1] = -1
        return x, y


def init_dataset(dataset: str):
    train_dataset = AbcDataset('train', dataset=dataset)
    test_dataset = AbcDataset('test', dataset=dataset)

    return train_dataset, test_dataset

def init_model(train_dataset):
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

def batch_end_callback(trainer):
    if trainer.iter_num % 100 == 0:
        print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")

def eval_add_split(model, trainer, train_dataset, test_dataset, split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length # naugy direct access shrug
    results = []
    mistakes_printed_already = 0
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    #loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
    for _, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)

        inp = x[:, :2*n]
        sol = y[:, -2*n:]
        
        cat = model.generate(inp, 2*n, do_sample=False) # using greedy argmax, not sampling
        sol_candidate = cat[:, -2*n:]         
        correct = (sol == sol_candidate).all(1).cpu() 
        for i in range(x.size(0)):
            results.append(int(correct[i]))
    
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

def run():
    train_dataset, test_dataset = init_dataset()
    model, trainer = init_model(train_dataset) 

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    model.eval()

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score = eval_add_split(model, trainer, train_dataset, test_dataset, 'train', max_batches=50)
        test_score  = eval_add_split(model, trainer, train_dataset, test_dataset, 'test',  max_batches=50)


# def random_long_mul_instance(length):
#     a = [random.randint(0,9) for i in range(length)]
#     b = [random.randint(0,9) for i in range(length)]
#
#     def str_c(t):
#         stage, max_c = t
#         s_c = str(stage)
#         return (max_c - len(s_c)) * '0' + s_c
#
#     val_a = int(''.join(str(d) for d in a))
#     val_b = int(''.join(str(d) for d in b))
#     val_c = val_a * val_b
#     
#     c_stage_1 = val_a * (val_b % 10)
#     c_stage_2 = val_a * ((val_b // 10) % 10)
#     c_stage_3 = val_a * ((val_b // 100) % 10)
#
#     res = ''.join(map(str_c, [(c_stage_1, 4), (c_stage_2*10, 5), (c_stage_3*100, 6), (val_c, 6)]))
#
#     return a + b + [int(d) for d in res]
#
# for i in range(10):
#     print (random_long_mul_instance(3))


# # In[20]:
#
#
# class MultLongDataset(Dataset):
#     """ 
#     Dataset for the Mult problem. E.g. for problem length 3:
#     12 * 333 = 3996
#     Input: 0 1 2 3 3 3 -> Output: 0 0 3 9 9 6
#     Which will feed into the transformer concatenated as:
#     input:  0 1 2 3 3 3 0 3 4
#     output: I I I 0 0 3 9 9 6
#     where I is "ignore", as the transformer is reading the input sequence
#     """
#
#     def __init__(self, split, length=3):
#         assert split in {'train', 'test'}
#         self.split = split
#         self.length = length
#     
#     def __len__(self):
#         return 20 # ...
#     
#     def get_vocab_size(self):
#         return 10
#     
#     def get_block_size(self):
#         # the length of the sequence that will feed into transformer, 
#         # containing concatenated input and the output, but -1 because
#         # the transformer starts making predictions at the last input element
#         return 2 * self.length + (((self.length + 1 + 2*self.length) * self.length) // 2) + 2 * self.length
#
#     def __getitem__(self, idx):
#         while True:
#             rai = random_long_mul_instance(self.length)
#             h = hash(str(rai[:2*self.length]))
#             
#             inp_split = 'test' if h % 4 == 0 else 'train' # designate 25% of examples as test
#             if inp_split == self.split:
#                 break # ok
#         
#         x = torch.tensor(rai[:-1], dtype=torch.long)
#         y = torch.tensor(rai[1:], dtype=torch.long)
#         
#         # we only want to predict at output locations, mask out the loss at the input locations
#         y[:2*self.length - 1] = -1
#         return x, y
#
#
# # In[1]:
#
#
# # print an example instance of the dataset
# train_dataset = MultLongDataset('train')
# test_dataset = MultLongDataset('test')
#
# for i in train_dataset:
#     print (i)
# x, y = train_dataset[0]
#
# print(len(train_dataset))
#
# print (x)
# for a, b in zip(x,y):
#     print(int(a),int(b))
#
#
# # In[22]:
#
#
# # create a GPT instance
#
# model_config = GPT.get_default_config()
# model_config.model_type = 'gpt-micro'
# #model_config.model_type = 'gpt-nano'
#
# model_config.vocab_size = train_dataset.get_vocab_size()
# model_config.block_size = train_dataset.get_block_size()
# model = GPT(model_config)
#
#
# # In[23]:
#
#
# print (model_config.n_head, model_config.n_layer, model_config.n_embd)
#
#
# # In[24]:
#
#
# # create a Trainer object
#
# train_config = Trainer.get_default_config()
# train_config.learning_rate = 5e-4 # the model we're using is so small that we can go a bit faster
# train_config.max_iters = 10000
# train_config.num_workers = 0
# trainer = Trainer(train_config, model, train_dataset)
#
#
# # In[25]:
#
#
# def batch_end_callback(trainer):
#     if trainer.iter_num % 100 == 0:
#         print(f"iter_dt {trainer.iter_dt * 1000:.2f}ms; iter {trainer.iter_num}: train loss {trainer.loss.item():.5f}")
# trainer.set_callback('on_batch_end', batch_end_callback)
#
# trainer.run()
#
#
# # In[26]:
#
#
# # now let's perform some evaluation
# model.eval()
# None
#
#
# # In[27]:
#
#
# def eval_long_mult_split(trainer, split, max_batches):
#     dataset = {'train':train_dataset, 'test':test_dataset}[split]
#     n = train_dataset.length # naugy direct access shrug
#     results = []
#     mistakes_printed_already = 0
#     loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
#     #loader = DataLoader(dataset, batch_size=1, num_workers=0, drop_last=False)
#     for b, (x, y) in enumerate(loader):
#         x = x.to(trainer.device)
#         y = y.to(trainer.device)
#
#         inp = x[:, :2*n]
#         sol = y[:, -2*n:]
#         
#         cat = model.generate(inp, (2*n + (((3*n + 1) * n) // 2)), do_sample=False) # using greedy argmax, not sampling
#         sol_candidate = cat[:, -2*n:]         
#         correct = (sol == sol_candidate).all(1).cpu() 
#         for i in range(x.size(0)):
#             results.append(int(correct[i]))
#     
#     rt = torch.tensor(results, dtype=torch.float)
#     print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
#     return rt.sum()
#
# # run a lot of examples from both train and test through the model and verify the output correctness
# with torch.no_grad():
#     train_score = eval_long_mult_split(trainer, 'train', max_batches=50)
#     test_score  = eval_long_mult_split(trainer, 'test',  max_batches=50)
