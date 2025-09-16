
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
from bpeasy.tokenizer import BPEasyTokenizer
from transformers import PreTrainedTokenizerFast

import torch.nn.utils.rnn as r
from datasets import load_dataset
import regex as re
import pandas as pd
import gzip
import json
import time
import random
import os


def shuffle_characters_in_strings(strings, proportion):
    # Step 1: Select a random proportion of strings to shuffle
    num_strings_to_shuffle = int(proportion * len(strings))
    indices_to_shuffle = random.sample(range(len(strings)), num_strings_to_shuffle)

    # Step 2: Shuffle the characters of the selected strings
    for idx in indices_to_shuffle:
        # Convert the string to a list of characters
        char_list = list(strings[idx])
        # Shuffle the characters
        random.shuffle(char_list)
        # Join the shuffled characters back into a string
        strings[idx] = ''.join(char_list)

    return strings

def memmap_arr(arr, dir):
    np_data = np.array(arr)
    memmap_file = np.memmap(dir + "/data.npy", dtype='uint16', mode='w+', shape=np_data.shape)
    memmap_file[:] = np_data
    memmap_file.flush()

def bpe_tokenize(vocab_size, shuffle=False, save_data=True):
    t = time.time()
    dataset = load_dataset("roneneldan/TinyStories", split='train').with_format("torch")
    texts = dataset['text']
    dataset_val = load_dataset("roneneldan/TinyStories", split='validation').with_format("torch")
    vals = dataset_val['text']

    # file_path = '/n/home03/tdatta/token-olmo/tank_dumps/medium_c4.json.gz'
    # with gzip.open(file_path, 'rt', encoding='utf-8') as f:
    #     ds = [json.loads(line) for line in f]
    # texts = [d['text'] for d in ds]
    # PROP_FOR_TRAIN = 
    # train = texts[int(PROP_FOR_TRAIN * len(texts)):]
    # vals = texts[:int(PROP_FOR_TRAIN * len(texts))]

    PROP_FOR_TRAIN = .2
    train = texts[:int(PROP_FOR_TRAIN * len(texts))]

    if shuffle:
        train = shuffle_characters_in_strings(train, 0.7)

    #do the tokenization
    regex_expression = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    tokenizer = BPEasyTokenizer.train(
            iter(train), # iterator over str
            vocab_size=vocab_size,
            max_token_length=16,
            regex_pattern=regex_expression,
            special_tokens=["<s>", "<pad>", "</s>"],
            fill_to_nearest_multiple_of_eight=True,
            name="bpeasy",
        )

    if save_data:
        if shuffle:
            directory_path = "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k_v2/bpe" + str(vocab_size) + "S"
        else:
            directory_path = "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k_v2/bpe" + str(vocab_size)

        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            os.makedirs(directory_path + "/train")
            os.makedirs(directory_path + "/val")

        tokenized_train = tokenizer.encode("<s>".join(texts), allowed_special={'<s>'})
        tokenized_val = tokenizer.encode("<s>".join(vals), allowed_special={'<s>'})

        memmap_arr(tokenized_train, directory_path + "/train")
        memmap_arr(tokenized_val, directory_path + "/val")

        #export the model
        tokenizer.export_to_huggingface_format(directory_path + "/bpe_tokenizer.json")
        tokenizer.save(directory_path + "/bpe_tokenizer_easy.json")

    #print the compression rate
    val = "<s>".join(vals)
    tokenized_texts = tokenizer.encode(val, allowed_special={'<s>'})
    compression_rate = len(val) / len(tokenized_texts)
    print("Compression rate of " + str(compression_rate) + " with vocab of " + str(vocab_size))
    print(time.time() - t)

if __name__ == "__main__":
    bpe_tokenize(7746)