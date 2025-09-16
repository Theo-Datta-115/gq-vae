from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch
import numpy as np
import regex as re
import torch.nn.utils.rnn as r
import gzip
import json
from datasets import load_dataset
import os 

"""
List of paths for data (I THINK THIS IS TOKENIZED WITH BOS/EOS!!!): 
- /n/holyscratch01/sham_lab/tokenae/regex10p : GPT2 regex
- /n/holyscratch01/sham_lab/tokenae/whitespace10p : whitespace regex
- /n/holyscratch01/sham_lab/tokenae/gpt_4_10p : GPT4 regex 
- /n/holyscratch01/sham_lab/tokenae/punct_10p : punct regex 
    c4_gpt2
"""

TRAIN_PERCENT = .9

def load_data(dataset, batch_size, partitions):
    if dataset == "tinystories":
        folder_path = '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/pre_data/ascii'
        files = [f for f in os.listdir(folder_path) if f.endswith('.pt')]
        files.sort() 
        data = [torch.load(os.path.join(folder_path, files[i])) for i in range(partitions)]
        data = torch.cat(data, dim=0)

        # data = data[:int(len(data) / 2)] # for partial data

        # data = torch.load('/n/holyscratch01/sham_lab/tokenae/regex10p.pt', weights_only=True)
        # data = torch.load('/n/holyscratch01/sham_lab/tokenae/tinystories_gpt2bytes.pt', weights_only=True)
    elif dataset == "c4":
        data = torch.load('/n/holyscratch01/sham_lab/tokenae/c4_gpt2.pt', weights_only=True)
    else:
        raise ValueError("Invalid dataset")

    training_data = data[:int(len(data) * TRAIN_PERCENT)]
    validation_data = data[int(len(data) * TRAIN_PERCENT):]

    td = SimpleDataset(training_data)
    vd = SimpleDataset(validation_data)
    
    #use collate_fn for non regex data (collate_fn=collate_fn)
    training_loader = DataLoader(td, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)
    validation_loader = DataLoader(vd, batch_size=batch_size, shuffle=True, pin_memory=True, drop_last=True)

    return training_data, validation_data, training_loader, validation_loader

def save_data(dataset, reg, i, split):
    use_ascii = False

    print("handling data")
    if dataset == "tinystories": 
        ds = load_dataset("roneneldan/TinyStories", split=('train' + split)).with_format("torch")
        text_column = ds['text']
    elif dataset == "c4":
        file_path = '/n/home03/tdatta/token-olmo/tank_dumps/medium_c4.json.gz'
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            ds = [json.loads(line) for line in f]
        text_column = [d['text'] for d in ds]
    else: 
        raise ValueError("Invalid dataset")

    text = "".join(text_column)
    regex = regex_rule(reg)
    words = re.findall(regex, text)
    if use_ascii:
        words = [word for word in words if len(word) <= 16]
        chrs = [torch.tensor([ord(char) for char in word if ord(char) < 128]) for word in words]
    else:
        words = [torch.tensor(list(word.encode('utf-8'))) for word in words]
        chrs = [word for word in words if len(word) <= 16]
    data = r.pad_sequence(chrs, batch_first=True)

    torch.save(data, '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/pre_data/bytes/' + dataset + '_' + reg + '-' + str(i) + '.pt')
    print('/n/netscratch/sham_lab/Everyone/tdatta/tokenae/pre_data/bytes/' + dataset + '_' + reg + '_' + str(i) + '.pt')

def regex_rule(normalization_rule_name: str) -> str:
    # GPT4 regex
    if normalization_rule_name == "gpt": #USED
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "gpt-num2":
        return r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # separates punctuation from words (except spaces)
    elif normalization_rule_name == "punct": #USED
        return r""" ?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # limits to 2 digits (use for vocab size < 50k to ensure full digit coverage)
    elif normalization_rule_name == "punct-num2":
        return r""" ?\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""
    # gpt 2 regex 
    elif normalization_rule_name == "gpt2": #USED
        return r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    # whitespace break
    elif normalization_rule_name == "whitespace": #USED
        return r'\s+'
    else:
        raise ValueError(f"Unknown normalization_rule_name {normalization_rule_name}")


class SimpleDataset(Dataset):
    def __init__(self, data):
        self.data = data
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

if __name__ == "__main__":
    data_segments = ["[:10%]", "[10:20%]", "[20:30%]", "[30:40%]", "[40:50%]", "[50:60%]", "[60:70%]", "[70:80%]", "[80:90%]", "[90:100%]"]
    for i, seg in enumerate(data_segments):
        save_data("tinystories", "gpt2", i, seg)


        
