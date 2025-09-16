"""
This is used for training BPE models and using them to tokenize data
Extra functionality:
 - save data into memmaped files, if PATH provided
 - allow you do permute the tokenizers to be trained on shuffled data
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, Dataset
import torch.nn.utils.rnn as r
from bpeasy.tokenizer import BPEasyTokenizer
from transformers import PreTrainedTokenizerFast
from datasets import load_dataset
import regex as re
import json
import time
import random
import os

# Helper function to shuffle the characters in strings, used to create a "weak BPE" baseline when shuffling BPE train data
def shuffle_characters_in_strings(strings, proportion):
    # Select a random proportion of strings to shuffle
    num_strings_to_shuffle = int(proportion * len(strings))
    indices_to_shuffle = random.sample(range(len(strings)), num_strings_to_shuffle)

    # Shuffle the characters of the selected strings
    for idx in indices_to_shuffle:
        char_list = list(strings[idx])
        random.shuffle(char_list)
        strings[idx] = ''.join(char_list)

    return strings

# Helper function to map data to memmap for fast access
def memmap_arr(arr, dir):
    np_data = np.array(arr)
    memmap_file = np.memmap(dir + "/data.npy", dtype='uint16', mode='w+', shape=np_data.shape)
    memmap_file[:] = np_data
    memmap_file.flush()

# Actual function to do BPE tokenization
def bpe_tokenize(vocab_size, shuffle=False, save_data=True):
    t = time.time() # Used for timing, for understanding speed constraints
    directory_path = "PATH" # path to save to, if save_data=True

    # Load the dataset, both train and val
    dataset = load_dataset("roneneldan/TinyStories", split='train').with_format("torch")
    texts = dataset['text']
    dataset_val = load_dataset("roneneldan/TinyStories", split='validation').with_format("torch")
    vals = dataset_val['text']

    # Proportion of train set used in the BPE training
    PROP_FOR_TRAIN = .2
    train = texts[:int(PROP_FOR_TRAIN * len(texts))]

    # Toggle for shuffling characters (including proportion to shuffle)
    if shuffle:
        train = shuffle_characters_in_strings(train, 0.7)

    # Use BPEasy to do the tokenization. Currently uses the GPT-2 Tokenizer
    # BPEasy used for speed, and to allow for easy use of custom regular expressions
    regex_expression = (r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""") 
    tokenizer = BPEasyTokenizer.train(
            iter(train),                                    # Iterator over the train set for the tokenizer
            vocab_size=vocab_size,                          # Vocabulary Size
            max_token_length=16,                            # Max token length, set to match GQ-VAE
            regex_pattern=regex_expression,                 # Follows the regular expression we set (gpt-2)
            special_tokens=["<s>", "<pad>", "</s>"],
            fill_to_nearest_multiple_of_eight=True,   
            name="bpeasy",
        )

    # toggle for saving the data
    if save_data:
        # Make directories
        if not os.path.exists(directory_path):
            os.makedirs(directory_path)
            os.makedirs(directory_path + "/train")
            os.makedirs(directory_path + "/val")

        # Encode both train and val, then save
        tokenized_train = tokenizer.encode("<s>".join(texts), allowed_special={'<s>'})
        tokenized_val = tokenizer.encode("<s>".join(vals), allowed_special={'<s>'})

        memmap_arr(tokenized_train, directory_path + "/train")
        memmap_arr(tokenized_val, directory_path + "/val")

        #export the model
        tokenizer.export_to_huggingface_format(directory_path + "/bpe_tokenizer.json")
        tokenizer.save(directory_path + "/bpe_tokenizer_easy.json")

    # Terminal outputs for compression rate and time
    val = "<s>".join(vals)
    tokenized_texts = tokenizer.encode(val, allowed_special={'<s>'})
    compression_rate = len(val) / len(tokenized_texts)
    print("Compression rate of " + str(compression_rate) + " with vocab of " + str(vocab_size))
    print(time.time() - t)

if __name__ == "__main__":
    bpe_tokenize(7746) # Set the vocab