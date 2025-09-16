import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tokenae import TokenAE
from datasets import load_dataset
from utils import SimpleDataset
from torch.utils.data import DataLoader

from transformers import PreTrainedTokenizer
import json
from tqdm import tqdm
import regex as re
import torch.nn.utils.rnn as r
import string
import argparse
import gzip

class LearnedTokenizer(PreTrainedTokenizer):
    def __init__(self,  model_save, process_vocab, vocab_fallback):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_save = torch.load(model_save)
        args = model_save['hyperparameters']
        self.args = args
        self.model = TokenAE(args['input_len'],args['codebook_dim'],args['embedding_dim'],args['n_embeddings'],args['kernel_size'],args['vocab_size'],args['alpha'],args['gamma'],args['beta']).to(self.device)
        self.model.load_state_dict(model_save['model'], strict=False)
        print("loaded!")
        self.model.eval()
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.EOS_token = '[EOS]'
        
        if not process_vocab:
            with open('vocab_regex.json', 'r') as f:
                vocab = json.load(f)
            with open('vocab_regex_map.json', 'r') as f:
                vocab_map = json.load(f)
        else:
            vocab, vocab_map = self.getDict()

        self.ids_to_tokens = {v: k for k, v in vocab.items()}
        self.vocab = vocab
        self.vocab_map = {int(k): v for k, v in vocab_map.items()}
        self.vocab_fallback = vocab_fallback
        self.lens_freq = []
        self.pred_lens_freq = []

        super().__init__(unk_token=self.unk_token, pad_token=self.pad_token)

    def encode(self, text):
        gpt2 = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        words = re.findall(gpt2, text)

        #THRESHOLD LENGTH
        words = [word for word in words if len(word) <= 16]
        words.append('                ') #ensure that there is a len 16 word
        chrs = [torch.tensor([ord(char) for char in word if ord(char) <= 128]) for word in words]
        x = r.pad_sequence(chrs, batch_first=True)
        x = x[:-1]

        #TODO Implement fallback here, using the whole model call 
        (recon_loss, _, _, _, _), x_hat, _, corr_token, gates, _, pred_mask_lens, masks, (min_encodings, correct_used_tokens, targets) = self.model(x, tokenizing=True, hardset=self.args['hardset'])
        # print("Encodign Compression is: " + str(corr_token[4].cpu().item()))
        
        mask_lens = ((masks > 0.5).float().sum(axis=2)).flatten()
        mask_lens_gated = mask_lens[gates.flatten() > 0.5]
        min_encodings_flattened = min_encodings[gates.flatten() > 0.5]
        pred_lens_flattened = ((pred_mask_lens.argmax(1) + 1).flatten())[gates.flatten() > 0.5]
        # self.lens_freq.append((min_encodings_flattened * mask_lens_gated.view(-1, 1)))
        # self.pred_lens_freq.append((min_encodings_flattened * pred_lens_flattened.view(-1, 1)))
        
        tokens = torch.argmax(min_encodings.reshape(-1, self.args['vocab_size']), dim=1).int()
        tokens = tokens[gates.flatten() > 0.5]

        #reduce to used vocab
        tokens = list(map(lambda x: self.vocab_map[x], tokens.tolist()))

        if self.vocab_fallback:
            incorrect = ~torch.all(correct_used_tokens, dim=1)
            # print("inc", torch.sum(incorrect))
            # print("inc %", torch.sum(incorrect) / len(incorrect))
            inc_tok = (targets * (masks > 0.5))[(gates > 0.5)][incorrect]
            replace = incorrect * (torch.arange(len(incorrect)) + 1).to(self.device)
            # print("arr", torch.arange(len(incorrect)))
            # print("rep", replace)
            replace = replace[replace != 0] - 1

            # print("rep", replace)
            for i, rep in enumerate(reversed(replace)):
                out = self.fallBack(inc_tok[len(replace) - i - 1])
                tokens[rep:rep+1] = out
    
        return tokens

    def encode_loader(self, loader, iters):
        chars = 0
        toks = 0
        tokens_data = []
        for i in tqdm(range(iters)):
            x = next(iter(loader)).to(self.device)
            (recon_loss, _, _, _, _), x_hat, _, corr_token, gates, _, _, masks, (min_encodings, correct_used_tokens, targets) = self.model(x, tokenizing=True, hardset=self.args['hardset'])
            tokens = torch.argmax(min_encodings.reshape(-1, self.args['vocab_size']), dim=1).int()
            tokens = tokens[gates.flatten() > 0.5]

            #reduce to used vocab
            tokens = list(map(lambda x: self.vocab_map[x], tokens.tolist()))

            if self.vocab_fallback:
                incorrect = ~torch.all(correct_used_tokens, dim=1)
                inc_tok = (targets * (masks > 0.5))[(gates > 0.5)][incorrect]
                replace = incorrect * torch.arange(len(incorrect)).to(self.device)
                replace = replace[replace != 0]

                for i, rep in enumerate(reversed(replace)):
                    out = self.fallBack(inc_tok[len(replace) - i - 1])
                    tokens[rep:rep+1] = out

            chars += torch.sum((x != 0).float()).item()
            toks += len(tokens)
            tokens_data.extend(tokens)

        print(chars / toks)
        return tokens_data

    def decode(self, token_ids):
        tok = self.convert_ids_to_tokens(token_ids)
        print(tok)
        return "".join(tok)

    def _tokenize(self, text):
        toks = self.encode(text)
        words = self.convert_ids_to_tokens(toks)
        return words

    def convert_tokens_to_ids(self, tokens, skip_special_tokens=False):
        # Convert tokens to IDs
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # Convert IDs back to tokens
        return [self.ids_to_tokens.get(i, self.unk_token) for i in ids]

    def save_vocabulary(self, save_directory, filename_prefix=None):
        # Save vocabulary as a JSON file
        vocab_file = f"{save_directory}/{filename_prefix}-vocab.json" if filename_prefix else f"{save_directory}/vocab.json"
        with open(vocab_file, 'w') as f:
            json.dump(self.vocab, f)
        return (vocab_file,)

    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        # Example: Add [CLS] at the beginning and [SEP] at the end
        return [self.vocab.get("[CLS]", 0)] + token_ids_0 + [self.vocab.get("[SEP]", 0)]

    def get_vocab(self):
        return self.vocab

    def getDict(self):
        emb = self.model.vector_quantization.embedding.weight.data
        emb_formatted = emb.unsqueeze(0)
        z, masks = self.model.decoder(emb_formatted)
        out = torch.argmax(z, axis=1)[0]
        masks = self.model.mask_approx_fun(masks)
        lens = (masks > 0.5).float().sum(axis=2)[0].int().tolist()
        # lens = mask_lens.argmax(1)[0]

        tokens = {}
        for i, tok in enumerate(out):
            tok = tok[-(lens[i]):]
            char_arr = [chr(o) for o in tok]
            token = "".join(char_arr)

            #remove extra post-word whitespace, if it accidentally exists
            token = re.sub(r'(?<=\S)\s+', '', token) 
            tokens[i] = token

        #Refine down any unused vocab
        unique_keys = set(tokens.values())
        print("keys len", len(unique_keys))
        unique_keys.update([chr(i) for i in range(0, 128)]) #adds fallback characters
        vocab = {key: value for value, key in enumerate(unique_keys)}
        vocab_map = {value: vocab[key] for value, key in enumerate(tokens.values())}

        print("total vocab:", len(vocab))

        #Add unk and pad
        vocab[self.unk_token] = len(vocab)
        vocab[self.pad_token] = len(vocab) + 1
        vocab[self.EOS_token] = len(vocab) + 2

        with open('vocab_regex.json', 'w') as json_file:
            json.dump(vocab, json_file, indent=2)
        with open('vocab_regex_map.json', 'w') as json_file:
            json.dump(vocab_map, json_file, indent=2)
        return vocab, vocab_map

    def fallBack(self, token_ids):
        token = "".join([chr(i) for i in token_ids[token_ids != 0]])
        token = re.sub(r'(?<=\S)\s+', '', token) #remove right side whitespace
        vocab = self.vocab.keys()

        result = []
        i = 0
        n = len(token)
        while i < n:
            # Find the longest substring that matches the current part of S
            max_len = 0
            chosen_substring = None
            
            for tok in vocab:
                if token.startswith(tok, i):
                    if len(tok) > max_len:
                        max_len = len(tok)
                        chosen_substring = tok

            if chosen_substring is None:
                print(token_ids)
                print(token)
                print("needs vocab")
                i += 1
            else:
                result.append(self.vocab[chosen_substring])
                i += max_len
    
        return result

    def checkVocab(self):
        vocab_initial = self.args['vocab_size']
        vocab_set = torch.load('unique_ids_' + str(vocab_initial) + ".pt")
        print('unique_ids_' + str(vocab_initial) + ".pt")
        vocab_reduced = torch.tensor(list(map(lambda x: self.vocab_map[x], vocab_set.tolist())))

        print(vocab_initial)
        print(len(vocab_set))
        print(len(torch.unique(vocab_reduced)))


def tokenize_dataset_loader(path):
    tokenizer = LearnedTokenizer(model_save=path, process_vocab=False, vocab_fallback=True)
    dataset = load_dataset("roneneldan/TinyStories", split='train[:1%]')
    
    text = "EOS".join(dataset['text'])
    regex = r"""EOS|'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    words = re.findall(regex, text)

    words = [word for word in words if len(word) <= 16]
    chrs = [torch.tensor([ord(char) for char in word if ord(char) < 128]) for word in words]
    data = r.pad_sequence(chrs, batch_first=True)

    chunk = 10000
    current = 0

    print(len(data) / chunk)
    for i in range(int(len(data) / chunk)):
        training_loader = DataLoader(SimpleDataset(data[current:current + chunk]), batch_size=1024, shuffle=True, drop_last=True)
        current += chunk
        tokens = tokenizer.encode_loader(training_loader, len(training_loader))
        print(tokens)

        np_data = np.array(tokens)
        memmap_file = np.memmap('/n/home03/tdatta/tank-vae/tokenized_data/tinystories' + str(i) + '.npy', dtype='uint16', mode='w+', shape=np_data.shape)
        memmap_file[:] = np_data
        memmap_file.flush()

"""
/n/netscratch/sham_lab/Everyone/tdatta/tokenae/2k_vocab/tinystories_tokenized
/n/netscratch/sham_lab/Everyone/tdatta/tokenae/2k_vocab/tinystories_bpe
"""


def tokenize_dataset(path, start):
    tokenizer = LearnedTokenizer(model_save=path, process_vocab=False, vocab_fallback=True)
    dataset = load_dataset("roneneldan/TinyStories", split='train')

    end = int((start + 0.1) * len(dataset)) #increments of 10% tokenization at once
    start = int(start * len(dataset))
    total = end - start
    dataset = dataset['text'][start: end]
    
    tokenized_data = []
    path = "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/hardset_3/train/"
    # tinystories_tokenized_val and tinystories_bpe_val

    data_lens = 0

    for i, data in enumerate(tqdm(dataset)):
        i = i + start
        if data != "":
            progress = 0

            # print(len(data))
            # print(f"Allocated memory: {torch.cuda.memory_allocated()/1024**2:.2f} MB")
            # print(f"Cached memory: {torch.cuda.memory_reserved()/1024**2:.2f} MB")
            data_lens += len(data)
            tokens = tokenizer.encode(data)
            tokenized_data.extend(tokens)
            tokenized_data.append(tokenizer.vocab['[EOS]'])

            # print(data_lens / len(tokenized_data))

            # if (i % (int(total / 5))) == 0:
            #     print("SAVING")
    #Write to Memmap
    np_data = np.array(tokenized_data)
    memmap_file = np.memmap(path + str(i) + '.npy', dtype='uint16', mode='w+', shape=np_data.shape)
    memmap_file[:] = np_data
    memmap_file.flush()

    #Reset data
    tokenized_data = []

def tokenize_dataset_val(path, data='tinystories'):
    tokenizer = LearnedTokenizer(model_save=path, process_vocab=False, vocab_fallback=True)
    print("VOCAB", len(tokenizer.vocab))
    
    if data != 'c4':
        dataset = load_dataset("roneneldan/TinyStories", split='validation')
        dataset = dataset['text']#[:400]
    else: 
        file_path = '/n/home03/tdatta/token-olmo/tank_dumps/medium_c4.json.gz'
        with gzip.open(file_path, 'rt', encoding='utf-8') as f:
            ds = [json.loads(line) for line in f]
        dataset = [d['text'] for d in ds][0:300]

    tokenized_data = []
    path = "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/hardset_3/val/"

    unique_set = set()
    data_lens = 0
    for i, data in enumerate(tqdm(dataset)):
        if data != "":
            # if len(data) > 2000:
            #     print("hi")
            #     data = data[:2000]
            data_lens += len(data)
            # print("__________")
            tokens = tokenizer.encode(data)
            tokenized_data.extend(tokens)
            tokenized_data.append(tokenizer.vocab['[EOS]'])
            # unique_set.update(torch.unique(torch.tensor(tokens)).tolist())
            # print(data_lens / len(tokenized_data))
            # print(tokenizer.decode(tokens))
            # print("used", len(tokens))
            # print("comp", len(data) / len(tokens))

    # lens_freq = torch.cat(tokenizer.lens_freq)
    # pred_lens_freq = torch.cat(tokenizer.pred_lens_freq)
    # print(torch.sum(pred_lens_freq != lens_freq))
    # print(torch.sum(pred_lens_freq != lens_freq) / len(tokenized_data))
    data = "".join(dataset)
    # print(len(data) / len(tokenized_data))
    # print("used tokens", len(torch.tensor(list(unique_set))))
    np_data = np.array(tokenized_data)
    memmap_file = np.memmap(path + 'data.npy', dtype='uint16', mode='w+', shape=np_data.shape)
    memmap_file[:] = np_data
    memmap_file.flush()

model_paths = [
    '/n/home03/tdatta/tank-vae/results/1024_gpt2/20000.pth', 
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_2k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_5k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_10k/20000.pth',
    '/n/home03/tdatta/tank-vae/results/1024_gpt2_20k/20000.pth']

model_paths_retry = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/10000.pth',
]

model_paths_c4 = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/5000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/10000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/15000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/20000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/2_50000/25000.pth'
]

model_paths_lr = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000difflr/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000difflr/25000.pth',
]

model_paths_lr_2 = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000ts_newschej2/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000tinystories/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000tinystories/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000ts_seed3/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000tinystories/35000.pth',
]

model_paths_perms = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000ts_d1024/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000mask_hardcode/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000maskpads_hardset_sweep/25000.pth'
]

model_paths_hardset = [
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_50000mask_hardcode/25000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_20000hardset_sweep/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_10000hardset_sweep/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_5000hardset_sweep/35000.pth',
    '/n/holyscratch01/sham_lab/tokenae/retry/saves/3_2000hardset_sweep/35000.pth'
]

model_paths_rotate = [
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/3_50000rot_trick/65000.pth',
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/4_50000rot_trick/65000.pth'
]

model_paths_5k = [
    '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/3_50000dim_tokens_4x/125000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/3_5000dim_sweep_long_512/65000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/3_5000dim_sweep_long_720/65000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/3_5000dim_sweep_long_1024/65000.pth',
    # '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000alpha_sweep_again/65000.pth',
]

if __name__ == "__main__":
    # for path in model_paths_5k:
    #     tokenize_dataset_val(path)

    # tokenizer = LearnedTokenizer(model_save = '/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_50000h8/20000.pth', process_vocab=True, vocab_fallback=True)
    # print("HARDSET", tokenizer.args['hardset'])
    # tokens = tokenizer.encode("I really hope that this tokenizes things well and has a good way of doing fallback!")
    # print(tokens)
    # text = tokenizer.decode(tokens)
    # print(text)


    parser = argparse.ArgumentParser(description="Process some files.")
    parser.add_argument('--start', type=str, default=0.0)
    parser.add_argument('--val', type=bool, default=False)
    args = parser.parse_args()
    # tokenizer = LearnedTokenizer(model_save='/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_20000h8_sweep/30000.pth', process_vocab=True, vocab_fallback=True)
    # toks = tokenizer.encode("hello, amazing, superb!")
    # print(toks)
    # print(tokenizer.decode(toks))

    if args.val:
        tokenize_dataset_val('/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_20000h8_sweep/30000.pth')
    else:
        print("START:", args.start)
        tokenize_dataset('/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/saves/2_20000h8_sweep/30000.pth', float(args.start))



# Example usage 
# tokenizer = LearnedTokenizer(model_save=model_paths[4], process_vocab=True, vocab_fallback=True)
# # tokenizer.checkVocab()
# tokens = tokenizer.encode("EOS")
# tokens = tokenizer.encode("Hi Varun, lets see how well this does on data that is out of distribution!")
# print(tokens)
# text = tokenizer.decode(tokens)
# print(text)


"""

dataset = load_dataset("roneneldan/TinyStories", split='train[:1%]')
for data in tqdm(dataset['text']):
    # print("___________________________________")
    
    # print(data)
    tokens = tokenizer.encode(data)
    # print(tokens)
    text = tokenizer.decode(tokens)
    # print(text)



CONSIDER DOING ANOTHER REGEX IN THE VOCAB TO PREVENT FOR TWO SPACES IN FRONT OF NON-SPACE CHARACTERS


/n/netscratch/sham_lab/Everyone/tdatta
"""
