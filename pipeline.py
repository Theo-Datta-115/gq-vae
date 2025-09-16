"""
This is a pipeline to check the compression rates of different variable distributions compared to BPE
"""

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Dataset
from utils import SimpleDataset, collate_fn
from models.tokenae import TokenAE
from tokenizers import Tokenizer, Regex
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Split, Whitespace
from regex_tokenizer import LearnedTokenizer
from data.data_compiled import load_data

from bpeasy.tokenizer import BPEasyTokenizer
from transformers import PreTrainedTokenizerFast

import torch.nn.utils.rnn as r
from datasets import load_dataset
import regex as re
import pandas as pd
import gzip
import json
import time
import argparse

from tqdm import tqdm

#hyperparams
BATCH_SIZE = 1024
LR = 1e-4
WEIGHT_DECAY = 1e-4

#static dimensions
EMBEDDING_DIM = 512
CODEBOOK_DIM = 512
N_EMBEDDINGS = 128
INPUT_LEN = 16
BETA = 0.25
GAMMA = 1
RS_SAMPLES = 131072

MAX_ITERS = 5000
VAL_ITERS = 100

#iterating dimensions
alphas = [1.5, 1.75, 2, 1, 1.25]
kernel_sizes = [10, 8]
vocab_sizes = [1000, 2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]

#whether to save
save = True
SAVE_MODEL_PATH = ""

#dataset
data = 'tinystories'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# change the seeding if doing multiple runs
torch.manual_seed(1)

#Regex to use
gpt_string = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
regex = re.compile(gpt_string)

def train(optimizer, training_loader, model):
    model.train()
    for i in range(MAX_ITERS):
        x = next(iter(training_loader)).to(device)
        optimizer.zero_grad()

        #train model on forward pass
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model(x, iter=i)

        #compute total loss from subloss
        loss = recon_loss + compression + mask_loss
        if codebook is not None and commitment is not None:
            loss += codebook + commitment 
        
        #backward pass
        loss.backward()
        optimizer.step()

        if (i % 50 == 0):
            print("i: " + str(i) + ",Correct Characters: " + str(corr_token[2]) + ", Compression: " + str(g_under_half))

def validate(val_loader, model, vocab=50000):
    model.eval()
    g = 0
    chars = 0
    inc_tok = 0
    inc_char = 0
    corrs = []
    recon_losses = []
    unique_set = set()
    for i in tqdm(range(VAL_ITERS)):
        x = next(iter(val_loader)).to(device)
        # print(x)
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks, (min_encodings, correct_used_tokens, targets) = model(x, iter=i, tokenizing=True)
        print(recon_loss)
            
        incorrect_tok = ~torch.all(correct_used_tokens, dim=1)
        incorrect_char = ~correct_used_tokens
        tokens = torch.argmax(min_encodings.reshape(-1, vocab), dim=1)
        unique_set.update(torch.unique(tokens).tolist())

        #Find the actual number of correct tokens among the tokens used
        
        corrs.append(corr_token[2].cpu().item())
        recon_losses.append((recon_loss).cpu().item())
        g += torch.sum((gates > 0.5).float()).item()
        chars += torch.sum((x != 0).float()).item()
        inc_tok += torch.sum(incorrect_tok).item()
        inc_char += torch.sum(incorrect_char).item()
    
    compression = chars / g
    corr = np.array(corrs).mean()
    loss_avg = np.array(recon_losses).mean()
    
    unique_ids = torch.tensor(list(unique_set))
    # torch.save(unique_ids, 'unique_ids_' + str(vocab) + ".pt")
    return compression, loss_avg, corr, (inc_tok / g), (inc_char / chars)

def train_compression_sweep():

    #Do data
    print("Loading Data")
    training_data, validation_data, training_loader, validation_loader = load_data(data, BATCH_SIZE)
    print("Training")
    for KERNEL_SIZE in kernel_sizes:
        for ALPHA in alphas:
            for VOCAB_SIZE in  vocab_sizes:
                # DO MODEL TRAINING
                model = TokenAE(INPUT_LEN, CODEBOOK_DIM, EMBEDDING_DIM, N_EMBEDDINGS, KERNEL_SIZE, VOCAB_SIZE, ALPHA, BETA, GAMMA, rs_samples=RS_SAMPLES).to(device)
                optimizer = optim.Adam(model.parameters(), lr=LR, amsgrad=True, weight_decay=WEIGHT_DECAY)
                train(optimizer, training_loader, model)

                if save:
                    torch.save(model.state_dict(), "/n/holyscratch01/sham_lab/tokenae/sweep_save/" + "a" + str(ALPHA) + "k" + str(KERNEL_SIZE) + "v" + str(VOCAB_SIZE) + '.pth')
                
                compression, loss_avg, corr = validate(validation_loader, model)
                
                #Create Dataframe for Saving
                new_data = pd.DataFrame({
                'Alpha': [ALPHA],
                'Kernel Size': [KERNEL_SIZE],
                'Vocab Size': [VOCAB_SIZE],
                'Compression': [compression],
                'Recon Loss': [loss_avg],
                'Correct %': [corr],
                })
                new_data.to_csv('1d_sweep.csv', mode='a', header=False, index=False)
                
                print(compression, loss_avg, corr)

def save_tinystories():
    dataset = load_dataset("roneneldan/TinyStories", split='train').with_format("torch")

    # Extract all texts from the dataset
    texts = dataset['text']
    text = "".join(texts)
    with open("tiny_stories.txt", "w", encoding="utf-8") as f:
        f.write(text)

def graph_models(model_paths):
    training_data, validation_data, training_loader, validation_loader = load_data(data, BATCH_SIZE, 1)
    for model_p in model_paths:
        model_save = torch.load(model_p)
        args = model_save['hyperparameters']
        model = TokenAE(args['input_len'],args['codebook_dim'],args['embedding_dim'],args['n_embeddings'],args['kernel_size'],args['vocab_size'],args['alpha'],args['gamma'],args['beta']).to(device)
        model.load_state_dict(model_save['model'], strict=False)
        compression, loss_avg, corr, inc_tok, inc_char = validate(validation_loader, model, vocab=args['vocab_size'])

        print("____________")
        print("Model Path: " + model_p)
        print(compression)
        print(loss_avg)
        print(corr)
        print(inc_tok)
        print(inc_char)
        
def regex_fallback(model_paths):
    for path in model_paths:
        ttraining_data, validation_data, training_loader, validation_loader = load_data(data, BATCH_SIZE, 1)
        tokenizer = LearnedTokenizer(model_save=path, process_vocab=True, vocab_fallback=True)
        tokenizer.encode_loader(validation_loader, 500)

model_paths = [
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_2k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_5k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_10k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_20k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2/20000.pth']

model_paths_retry = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000/15000.pth',
    # '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/10000.pth',
]

model_paths_lr = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000difflr/25000.pth',
]

model_paths_lr_2 = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000ts_newschej2/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000tinystories/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000tinystories/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000ts_seed3/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000tinystories/35000.pth',
]

model_paths_hardset = [
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000gates2_hardset/20000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_20000gates2_hardset/25000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_1000gates2_hardset/25000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_2000gates2_hardset/25000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_5000gates2_hardset/25000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_10000gates2_hardset/25000.pth',

]

model_paths_hardset_paper = [
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_2000h8_sweep/40000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h10/20000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h9/20000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h8/20000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h7/20000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h6/20000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h5/20000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h4/20000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h3/5000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h2/5000.pth',
]

model_paths_hardset_8 = [
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h8_sweep/20000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_20000h8_sweep/30000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_10000h8_sweep/40000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_5000h8_sweep/40000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_2000h8_sweep/40000.pth',
]

# regex_fallback(model_paths)
# graph_models(model_paths)


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description="Process some files.")
    # parser.add_argument('--path', type=str, default="")
    # parser.add_argument('--destination', type=str, default="")
    # args = parser.parse_args()
    # bpe_tokenize_variable(args.path, args.destination)
    # bpe_tokenize()
    # bpe_tokenize_val()
    # sweep_bpe(vocab_sizes)
    # data = np.memmap("/n/holyscratch01/sham_lab/tokenae/tinystories_bpe_val/data.npy", dtype='uint16', mode='r')
    # print(data[0:1000])
    # graph_models(model_paths_lr_2)
    # print(np.array(10))
    regex_fallback(reversed(model_paths_hardset_paper))
    

# save_tinystories()
# train_compression_sweep()
# sweep_bpe(vocab_sizes)


"""

                tokenizer = Tokenizer(BPE())
                tokenizer.pre_tokenizer = Split(pattern=gpt_string, behavior="isolated")
                trainer = BpeTrainer(vocab_size=VOCAB_SIZE, min_frequency=2, special_tokens=["<s>", "<pad>", "</s>", "<unk>", "<mask>"])
                tokenizer.train(['tiny_stories1p.txt'], trainer)
                encoded = tokenizer.encode(text)
                print(encoded)
                print(len(text))
                print(len(encoded))


/n/holyscratch01/sham_lab/tokenae/2k_vocab/tinystories_bpe/data.npy
bpe_tokenizer-2k.json

/n/holyscratch01/sham_lab/tokenae/tinystories_bpe/data.npy
bpe_tokenizer-10k-eos.json
"""