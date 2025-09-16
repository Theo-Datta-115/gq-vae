import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.tokenae import TokenAE
from data.load_data import LoadData
from data.synthetic import Synthetic

from transformers import PreTrainedTokenizer
import re
import json
from tqdm import tqdm
import regex
import torch.nn.utils.rnn as r

class LearnedTokenizer(PreTrainedTokenizer):
    def __init__(self,  model_save, process_vocab, vocab_fallback, regex=True, vocab_file='/n/home03/tdatta/tank-vae/vocab_regex.json'):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_save = torch.load(model_save)
        args = model_save['hyperparameters']
        self.model = TokenAE(args['input_len'],args['codebook_dim'],args['embedding_dim'],args['n_embeddings'],args['kernel_size'],args['vocab_size'],args['alpha'],args['gamma'],args['beta']).to(self.device)
        self.model.load_state_dict(model_save['model'], strict=False)
        self.unk_token = '[UNK]'
        self.pad_token = '[PAD]'
        self.regex = regex
        
        if not process_vocab:
            with open(vocab_file, 'r') as f:
                ids_to_tokens = json.load(f)
        else:
            ids_to_tokens = self.getDict('vocab_regex.json')

        self.ids_to_tokens = {int(k): v for k, v in ids_to_tokens.items()}
        self.vocab = {v: k for k, v in self.ids_to_tokens.items()}

        super().__init__(unk_token=self.unk_token, pad_token=self.pad_token)

    def encode(self, text):
        if self.regex:
            return self.encode_regex(text)
        else:
            return self.encode_normal(text)

    def encode_normal(self, text):
        # Tokenization logic (e.g., character-level splitting)
        x = torch.tensor([ord(char) for char in text if ord(char) < 128])
        x = self.reshape_and_pad(x)
        xl = x.long().to(self.device)
        x_embed = self.model.embedding_layer(xl).float()
        z_e, g_enc = self.model.encoder(x_embed)
        _, z_q, _, min_encodings, _ = self.model.vector_quantization(z_e)
        gates, _, _ = self.model.gater(z_q)

        tokens = torch.argmax(min_encodings.reshape(-1, 50000), dim=1).int()
        tokens = tokens[gates.flatten() > 0.5]

        #CHECKING INTERNALS
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = self.model(x,verbose=True, stage='test')
        return tokens.tolist()

    def encode_regex(self, text):
        gpt2 = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        words = regex.findall(gpt2, text)

        #THRESHOLD LENGTH
        words = [word for word in words if len(word) <= 14]
        chrs = [torch.tensor([ord(char) for char in word if ord(char) <= 128]) for word in words]
        x = r.pad_sequence(chrs, batch_first=True)

        xl = x.long().to(self.device)
        x_embed = self.model.embedding_layer(xl).float()
        z_e, g_enc = self.model.encoder(x_embed)
        _, z_q, _, min_encodings, _ = self.model.vector_quantization(z_e)
        gates, _, _ = self.model.gater(z_q)

        tokens = torch.argmax(min_encodings.reshape(-1, 50000), dim=1).int()
        tokens = tokens[gates.flatten() > 0.5]

        return tokens.tolist()

    def _tokenize(self, text):
        toks = self.encode(text)
        words = self.convert_ids_to_tokens(toks)
        return words

    def convert_tokens_to_ids(self, tokens, skip_special_tokens=False):
        # Convert tokens to IDs
        return [self.vocab.get(token, self.vocab[self.unk_token]) for token in tokens]

    def convert_ids_to_tokens(self, ids, skip_special_tokens=False):
        # Convert IDs back to tokens
        for id in ids:
            z = self.model.vector_quantization.embedding.weight.data[id]
            out, _ = self.model.decoder(z.reshape(1,1,512))
            output_ints = torch.argmax(out.reshape(1,128,1 * 10), axis=1).reshape(1, 10)
            token = "".join(["".join([chr(i) for i in s]) for s in output_ints])
            print("__________")
            print(id)
            print(token)
            print(self.ids_to_tokens.get(id, self.unk_token))
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

    def reshape_and_pad(self, tensor, target_rows=64, pad_value=3):
        flattened_tensor = tensor.flatten()
        to_pad = (int(len(flattened_tensor) / 64) + 1) * 64 - len(flattened_tensor)
        flattened_tensor = F.pad(flattened_tensor, (0, to_pad), value=pad_value)
        return flattened_tensor.view(int(len(flattened_tensor) / target_rows), target_rows)

    def getfreqencies(self):
        NUM_ITER = 100
        if self.regex:
            training_data, validation_data, training_loader, validation_loader = Synthetic(256, 1024000, 16, .9, usePregen=True)
        else:
            training_data, validation_data, training_loader, validation_loader = LoadData('tinystories', 256, 64, .9)
        self.model.eval()
        len_freqs = torch.zeros(50000, 10).to(self.device)
        print("Getting frequencies of different latent lengths")
        for i in tqdm(range(NUM_ITER)):
            x = next(iter(training_loader))   
            (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks, (mask_lens, min_encodings) = self.model(x, verbose=False, stage='test', tokenizing=True)

            mask_lens = (mask_lens - 1).int()
            min_encoding_indices = torch.argmax(min_encodings, dim=1).int()

            for (loc, len) in zip(min_encoding_indices, mask_lens):
                len_freqs[loc][len] += 1

        return len_freqs

    def getDict(self, saveLink):
        freq = self.getfreqencies()
        lens = freq.argmax(1) + 1

        emb = self.model.vector_quantization.embedding.weight.data
        emb_formatted = emb.unsqueeze(0)
        z, _ = self.model.decoder(emb_formatted)
        out = torch.argmax(z, axis=1)[0]

        tokens = {}
        for i, tok in enumerate(out):
            tok = tok[-lens[i]:]
            char_arr = [chr(o) for o in tok if o != 2 and o != 3]
            token = "".join(char_arr)
            tokens[i] = token

        tokens[50001] = self.unk_token
        tokens[50002] = self.pad_token

        with open(saveLink, 'w') as json_file:
            json.dump(tokens, json_file, indent=2)
        return tokens

# Example usage
tokenizer = LearnedTokenizer(model_save='/n/home03/tdatta/tank-vae/results/regex/25000.pth', process_vocab=False, vocab_fallback=True)
tokens = tokenizer.encode("One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp.")
print(tokens)
text = tokenizer.decode(tokens)
print(text)