# GENERAL IMPORTS
import numpy as np
import torch
import torch.nn as nn
import json
import regex
from datasets import load_dataset

import sys
# user = 'kayla'
user = 'theo'

if user == 'kayla':
    ROOT = '/n/home02/klhuang/tank-vae/'
    sys.path.append('/n/home02/klhuang/tank-vae/')
    PATH = ROOT + 'results/old-loss-len/epoch1.pth'
else:
    ROOT = '/n/home03/tdatta/tank-vae/'
    sys.path.append('/n/home03/tdatta/tank-vae/')
    PATH = ROOT + 'results/linear_decoder/6.pth'

# PATH = '/n/home03/tdatta/tank-vae/results/vqvae_data_64_512_512_128_10_5000_1.5_0.25_.pth'
PATH = '/n/home03/tdatta/tank-vae/results/50k_nolen/30000.pth'
PATH = '/n/home03/tdatta/tank-vae/results/regex/26000.pth'

# CUSTOM IMPORTS
from models.tokenae import TokenAE
from data.load_data import LoadData
from data.synthetic import Synthetic

# LOADING MODEL
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_save = torch.load(PATH)

# CAREFUL: format for your inputs
#model = TokenAE(args.input_len, args.codebook_dim, args.embedding_dim, args.n_embeddings, args.kernel_size, args.vocab_size, args.alpha, args.beta).to(device)
model = TokenAE(16,512,512,128,10,50000,1.5,0.25,1).to(device)
model.load_state_dict(model_save['model'], strict=False)

def testRun(x_use=0, usex=False):
    # x = "let us all hope that we can tokenize stuff easily, abracadabra!"
    x = "beautiful"
    # x = "One day, a little girl named Lily found a needle in her room."
    # x = "how can I tell if this model is learning real outputs?"
    
    # WARNING: BOS AND EOS might be different, please take care
    
    x_d = [2] + [ord(c) for c in x] + [0 for _ in range(14 - len(x))] + [3] 
    x = torch.reshape(torch.Tensor(x_d), (1, -1)).to(device)
    
    if usex:
        x = x_use

    model.eval()
    (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model(x, verbose=False, stage='test')

    # min_encoding_indices = torch.argmax(min_encodings, dim=1).int()
    # emb = model.vector_quantization.embedding.weight.data[min_encoding_indices]
    # emb_formatted = emb.unsqueeze(0)
    # x_hat, _ = model.decoder(emb_formatted)

    print(recon_loss, codebook, commitment, compression, mask_loss)
    # print("predicted", pred_masks)
    print("real", masks > 0.5)
    print("".join([chr(i) for i in (x[0].int().tolist())]))
    print("\033[95minputs", "".join([chr(i) for i in x[0].int().tolist()]), "\033[0m")
    
    output_ints = torch.argmax(x_hat.reshape(1,128,len(x_d) * 10), axis=1).reshape(len(x_d), 10)
    # print("tokens", output_ints)
    output_ints_tensor = output_ints.flatten()

    masks_binary = torch.where(masks > 0.5, 1, 0)
    masks_flat = masks_binary.flatten()
    
    gates[:,-1] = 1
    gates = (gates > 0.5).float()
    print("gates", gates)
    # print("pred masks", pred_masks[gates > 0.5])
    gates2 = gates.unsqueeze(2).expand(-1, -1, 10)
    gates2.flatten()
    real = (gates2.flatten().bool() & masks_flat.bool()).to(device)
    output_ints_tensor_masked = output_ints_tensor[real].tolist()

    char_arr = [chr(i) for i in output_ints_tensor_masked]
    all_chars = "".join(char_arr)
    print("\033[92moutputs: ", all_chars, "\033[0m")

    total_tokens = output_ints[gates.reshape(len(x_d),) > 0.5]
    tokens = "\n".join(["".join([chr(i) for i in s]) for s in total_tokens])
    print("\033[93moutputs tokens: ", tokens, "\033[0m")


#this currently outputs nonsense, but it might be because it needs to see EOS and BOS???
def testWord():
    i = 0
    
    # try adding the BOS and EOS??
    emb = model.vector_quantization.embedding.weight.data
    emb_formatted = emb.unsqueeze(0)
    z, lens = model.decoder(emb_formatted)
    # lens = model.mask_decoder(emb_formatted, "gates")
    out = torch.argmax(z, axis=1)[0]
    lens = torch.argmax(lens, axis=1)[0]
    for idx in range(len(lens)):
        o = out[idx]
        l = 9#lens[idx]
        char_arr = [chr(o[9 - i]) for i in range(l)]
        if len(char_arr) != 0 and char_arr[0] == " ":
            all_chars = "".join(char_arr)
            print(all_chars)

    print(out.shape)
    # output = model.decoder(e)
    # print(output.shape)

    # print(intToChar(torch.argmax(dec, axis=1)[0].tolist()))

def testInternal():
    x = "how can I tell if this model is learning real outputs?"
    
    # WARNING: BOS AND EOS might be different, please take care
    x_d = [2] + [ord(c) for c in x] + [3] 
    x = torch.reshape(torch.Tensor(x_d), (1, -1)).to(device)

    xl = x.long().to(model.embedding_layer.weight.device)
    x_embed = model.embedding_layer(xl)
    # x_perm = x_embed.permute(0,2,1).contiguous().float()
    z_e = model.encoder(x_perm).to(device)

    (codebook, commitment), z_q, codebook_used, min_encodings, d = model.vector_quantization(z_e, verbose=False)
    gates, compression, g_under_half = model.gater(z_q)

    print(z_q.shape)
    print(model.vector_quantization.embedding.weight.data.shape)

    # for z in z_q[0].permute(1,0).contiguous():
    for z in model.vector_quantization.embedding.weight.data:
        out = model.decoder(z.reshape(1,512,1))
        output_ints = torch.argmax(out.reshape(1,128,1 * 10), axis=1).reshape(1, 10)
        tokens = "".join(["".join([chr(i) for i in s]) for s in output_ints])
        print("\033[93moutputs tokens: ", tokens, "\033[0m")

    x_hat = model.decoder(z_q)

    output_ints = torch.argmax(x_hat.reshape(1,128,len(x_d) * 10), axis=1).reshape(len(x_d), 10)
    total_tokens = output_ints[gates.reshape(len(x_d),) > 0.5]
    tokens = "\n".join(["".join([chr(i) for i in s]) for s in total_tokens])
    print("\033[93moutputs tokens: ", tokens, "\033[0m")

def testLens():
    training_data, validation_data, training_loader, validation_loader = LoadData('tinystories', 256, 64, .9)
    
    model.eval()

    len_freqs = torch.zeros(50000, 10).to(device)

    for i in range(1000):
        x = next(iter(training_loader))   
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks, (mask_lens, min_encodings) = model(x, verbose=False, stage='test')

        mask_lens = (mask_lens - 1).int()
        min_encoding_indices = torch.argmax(min_encodings, dim=1).int()

        # len_freqs[min_encoding_indices][mask_lens] += 1
        for (loc, len) in zip(min_encoding_indices, mask_lens):
            len_freqs[loc][len] += 1
        print(i)

    torch.save(len_freqs, 'model_lens_freq.pt')

    #GOAL: I want to cycle through a large portion of the data (ex. 10,000 forward passes) and compile the average lengths of each 

def getDict():
    freq = torch.load('/n/home03/tdatta/tank-vae/testing/model_lens_freq.pt')
    # lens_underused = freq.sum(1) < 10
    lens = freq.argmax(1) + 1
    # lens[lens_underused] = 1

    # try adding the BOS and EOS??
    emb = model.vector_quantization.embedding.weight.data
    emb_formatted = emb.unsqueeze(0)
    z, _ = model.decoder(emb_formatted)
    # lens = model.mask_decoder(emb_formatted, "gates")
    out = torch.argmax(z, axis=1)[0]

    print(out.shape)
    tokens = {}
    for i, tok in enumerate(out):
        tok = tok[-lens[i]:]
        char_arr = [chr(o) for o in tok if o != 2 and o != 3]
        token = "".join(char_arr)
        tokens[i] = token

    tokens[50001] = "[UNK]"
    tokens[50002] = "[PAD]"

    with open('vocab.json', 'w') as json_file:
        json.dump(tokens, json_file, indent=2)

    # for i in freq:
    #     print(i)

def testRegex():
    training_data, validation_data, training_loader, validation_loader = Synthetic(256, 102400, 16, .9, usePregen=True)
    g = 0
    chars = 0
    corr = 0
    count = 0
    
    for x_b in training_loader:
        x = x_b.to(device)
        model.eval()
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model(x, verbose=False, stage='test')

        g += torch.sum((gates > 0.5).float()).item()
        chars += torch.sum((x != 0).float()).item()
        count += len(x)
        print("compression", chars / g)
        print("skyline", chars / count)
        # print("prop filler", torch.sum((x == 0).float()).item() / (256 * 16))

def seeDataRegex():
    training_data, validation_data, training_loader, validation_loader = Synthetic(256, 102400, 16, .9, usePregen=True)
    for x_b in training_loader:
        x_b = x_b.to(device)
        for x in x_b:
            x = x.unsqueeze(0)
            testRun(x, usex=True)

# testRun()
# testRegex()
testRegex()
# testInternal()
# testLens()
# getDict()



# def testRegexCompression():
#     PATH2 = '/n/home03/tdatta/tank-vae/results/pleasesave/10000.pth'
#     model_save2 = torch.load(PATH)
#     model2 = TokenAE(64,512,512,128,10,50000,1.5,0.25,1).to(device)
#     model2.load_state_dict(model_save2['model'], strict=False)

#     ds = load_dataset("roneneldan/TinyStories", split=f'train[:1%]').with_format("torch")
#     gpt2 = regex.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    
#     text_column = ds['text']
#     for string in text_column:
#         unicode_values = [ord(char) for char in string]
#         data2 = torch.tensor(unicode_values, dtype=torch.int32)
#         data2 = data2[data2 < 128]
#         data2 = data2.to(device)

#         (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model2(data2, verbose=False, stage='test')

#         tokens = torch.sum((gates > 0.5).float())

#         print("RECON NOREGEX", recon_loss)
#         print("TOKENS NOREGEX", tokens)

#         words = regex.findall(gpt2, string)
#         words = [word for word in words if len(word) <= 14]
#         chrs = [torch.tensor([ord(char) for char in word if ord(char) <= 128]) for word in words]
#         data1 = r.pad_sequence(chrs, batch_first=True)

#         (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model1(data1, verbose=False, stage='test')

#         tokens = torch.sum((gates > 0.5).float())

#         print("RECON REGEX", recon_loss)
#         print("TOKENS REGEX", tokens)

