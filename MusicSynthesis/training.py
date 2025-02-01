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

set_seed(3407)
print("Loading trainer")

class AbcTokenizer:
    max_note = 25

    abc_julia_arr = [
        'Song',
        'Bar',
    ] + \
    [f'Note {i}' for i in range(-max_note, max_note + 1)] + \
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

    def decode(self, tokens: list[int]) -> str:
        return "\n".join([self.token_to_abc[tok] for tok in tokens])

class AbcDataset(Dataset):
    def __init__(self, split: str, dataset: str, length: int = 30):
        assert split in {'train', 'test'}
        self.split = split
        self.dataset = dataset
        self.datafiles = [d for d in os.listdir(dataset) if d.endswith(".abc")]
        self.datafiles_len = len(self.datafiles)
        self.sizes = {
            'train': self.datafiles_len - (self.datafiles_len // 4), 
            'test': self.datafiles_len // 4
        }

        self.tokenizer = AbcTokenizer()
        self.length = length
        self.split_part = 3

        self.data = self.load_data()

        random.shuffle(self.data)
    
    def __len__(self):
        return 20000 # ...

    def load_data(self):
        data = []
        for file in self.datafiles:
            with open(f"{self.dataset}/{file}", "r") as f:
                data.append(self.tokenizer.encode(f.read()))
        return data
    
    def get_vocab_size(self):
        return self.tokenizer.get_vocab()
    
    def get_block_size(self):
        # the length of the sequence that will feed into transformer, 
        # containing concatenated input and the output, but -1 because
        # the transformer starts making predictions at the last input element
        return self.length - self.split_part

    def get_inp(self, idx: int) -> int:
        match self.split:
            case 'train':
                return idx + (idx // 3)
            case 'test':
                return (idx + 1) * 4 - 1
        return 0

    def get_data(self, idx: int) -> list[int]:
        size = self.sizes[self.split]
        data_idx = self.get_inp(idx % size)

        return self.data[data_idx]

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

    def __getitem__(self, _):
        while True:
            rai = self.get_data(random.randint(0, self.datafiles_len + 1))
            if len(rai) - self.length < 0:
                continue

            random_first = random.randint(0, len(rai) - self.length)
            break
        
        x = torch.tensor(rai[random_first:random_first+self.length-self.split_part], dtype=torch.long)
        y = torch.tensor(rai[random_first+self.split_part:random_first+self.length], dtype=torch.long)

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

def music_split(model, trainer, train_dataset, test_dataset, split, max_batches):
    dataset = {'train':train_dataset, 'test':test_dataset}[split]
    n = train_dataset.length - train_dataset.split_part # naugy direct access shrug
    results = []
    loader = DataLoader(dataset, batch_size=100, num_workers=0, drop_last=False)
    for _, (x, y) in enumerate(loader):
        x = x.to(trainer.device)
        y = y.to(trainer.device)

        inp = x[:, :n//2 + 1]
        sol = y[:, -n//2:]
        
        cat = model.generate(inp, n//2, do_sample=True)
        sol_candidate = cat[:, -n//2:]
        # inny sposób oceniania
        correct = (sol == sol_candidate).all(1).cpu() 
        for i in range(x.size(0)):
            results.append(int(correct[i]))
    
    rt = torch.tensor(results, dtype=torch.float)
    print("%s final score: %d/%d = %.2f%% correct" % (split, rt.sum(), len(results), 100*rt.mean()))
    return rt.sum()

def run():
    train_dataset, test_dataset = init_dataset("../run2025-02-0116-24-01")
    model, trainer = init_model(train_dataset) 

    trainer.set_callback('on_batch_end', batch_end_callback)
    trainer.run()
    model.eval()

    # run a lot of examples from both train and test through the model and verify the output correctness
    with torch.no_grad():
        train_score = music_split(model, trainer, train_dataset, test_dataset, 'train', max_batches=50)
        test_score  = music_split(model, trainer, train_dataset, test_dataset, 'test',  max_batches=50)

    torch.save(model.state_dict(), f"model/model{datetime.now().strftime("%Y-%m-%d-%H-%M-%S")}")

def test():
    train_dataset, test_dataset = init_dataset("../run2025-02-0115-35-50")
    model, trainer = init_model(train_dataset)

    model.load_state_dict(torch.load("model/model2025-02-01-15-53-21"))
    model.eval()
    
    with open("../run2025-02-0115-35-50/120.abc") as test_music_f:
        test_music = train_dataset.tokenizer.encode(test_music_f.read())

    test_music = torch.tensor(test_music, dtype=torch.long).to(trainer.device)
    test_music = test_music[None, :]
    cat = model.generate(test_music, 50, do_sample=True)

    print(train_dataset.tokenizer.decode(cat.cpu().detach().numpy()[0]))

if __name__ == "__main__":
    run()
