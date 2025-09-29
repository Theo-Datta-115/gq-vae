import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import os
from data.data_compiled import load_data
import time
import wandb
import os
import yaml
import pprint
from types import SimpleNamespace
from models.tokenae import TokenAE
from torch.profiler import profile, record_function, ProfilerActivity
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from utils import CosWithWarmup

# from line_profiler import LineProfiler
## FLAGS
parser = argparse.ArgumentParser()

os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()

"""
Hyperparameters
"""
# make timestamp readable
timestamp = time.ctime().replace('  ', ' ').replace(' ', '_').replace(':', '_').lower()

#training params
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--epochs", type=int, default=15)
parser.add_argument("--learning_rate", type=float, default=1e-4) #was 5e-5
parser.add_argument("--log_interval", type=int, default=50)
parser.add_argument("--weight_decay", type=float, default=1e-4)

#model internal setting
parser.add_argument("--embedding_dim", type=int, default=1024) #dimension the characters are embedded to (E)
parser.add_argument("--n_embeddings", type=int, default=128) #number of unique characters in your dataset: 256 for C4 and 128 for tinystories
parser.add_argument("--input_len", type=int, default=16) #length of input sequences (T)
parser.add_argument("--codebook_dim", type=int, default=1024) #codebook dimensionality (D)
parser.add_argument("--alpha", type=float, default=3) #loss term for compression
parser.add_argument("--beta", type=float, default=.25) #loss term for commitment in the codebook
parser.add_argument("--gamma", type=float, default=1) #loss term for the mask loss
parser.add_argument("--vocab_size", type=int, default=50000) #number of vectors in the codebook (V)
parser.add_argument("--kernel_size", type=int, default=10) #kernel to use (window)
parser.add_argument("--hardset", type=int, default=None) #kernel to use (window)

# data settings
parser.add_argument("--data", type=str, default="tinystories")
parser.add_argument("--seed", type=int, default=1)

# whether or not to save model, wandb
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--filename",  type=str, default=timestamp)
parser.add_argument("--use_wandb",  type=bool, default=False)
parser.add_argument("--name", type=str, default="default")

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
run_name = args.name

if args.use_wandb:
    wandb.init(project='test_tokenizer', entity='harvardml', name=(run_name + str(args.alpha) + "v" + str(args.vocab_size)))
    wandb.config = {
        "learning_rate": args.learning_rate,
        "epochs": args.epochs,
        "batch_size": args.batch_size
    }

def save(path, results, n_updates):
    #chceck if the dict exists, if not create it
    if not os.path.exists(path):
        os.makedirs(path)

    path = path + "/" + str(n_updates) + '.pth'
    torch.save(results, path)

def log(results, logging):
    if results["n_updates"] % args.log_interval == 0:
        print("Update #", results["n_updates"], end=": ")
    for i, key in enumerate(list(results.keys())[1:]):
        if (logging[i] is not None):
            if torch.is_tensor(logging[i]):
                results[key].append(logging[i].cpu().detach().numpy())
            else:
                results[key].append(round(logging[i], 5))
        if results["n_updates"] % args.log_interval == 0:
            print(key, ":", np.mean(results[key][-args.log_interval:]), end=", ")
            if args.use_wandb:
                wandb.log({key: np.mean(results[key][-args.log_interval:])})
    results["n_updates"] += 1
    if results["n_updates"] % args.log_interval == 0:
        print()
    return results

def train(optimizer, scheduler, training_loader, validation_loader, model, results, val_results, SAVE_PATH="PATH"):
    model.train()
    for x in training_loader:
        x = x.to(device)
        optimizer.zero_grad()

        # Run the model
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model(x, iter=results['n_updates'], hardset=args.hardset)
        
        # Calculate the joint loss
        loss = recon_loss + compression + mask_loss
        if codebook is not None and commitment is not None:
            loss += codebook + commitment 

        loss.backward()
        optimizer.step()
        scheduler.step()

        # If needed, save the model and the run data
        if args.save and (results['n_updates'] % 5000 == 0):
            if SAVE_PATH == "PATH":
                print("Please input a path for model saves, if using args.save, as SAVE_PATH")
            path = SAVE_PATH
            results_to_save = {'model': model.state_dict(), 'results': results, 'hyperparameters': args.__dict__}
            save(path, results_to_save, results['n_updates'])

        if (results['n_updates'] % 1001 == 0) and (results['n_updates'] != 0):
            validate(validation_loader, model, val_results)

        # Log everything (both in console, and if needed, in wandb)
        logging = [loss, recon_loss, codebook, commitment, compression, mask_loss, g_under_half, codebook_used, corr_token[0], corr_token[1], corr_token[2], corr_token[3], corr_token[4], optimizer.param_groups[0]['lr']]
        results = log(results, logging)

def validate(validation_loader, model, val_results):
    # Do 50 forward passes as a val 
    for i in range(50):
        x = next(iter(validation_loader)).to(device)
        (recon_loss, codebook, commitment, compression, mask_loss), x_hat, codebook_used, corr_token, gates, g_under_half, pred_masks, masks = model(x, iter=results['n_updates'], hardset=args.hardset)
        loss = recon_loss + compression + mask_loss
        if codebook is not None and commitment is not None:
            loss += codebook + commitment 

        logging = [loss, recon_loss, mask_loss, codebook_used, corr_token[2], corr_token[3], corr_token[4]]
        val_results = log(val_results, logging)
        
if __name__ == "__main__":
    training_data, validation_data, training_loader, validation_loader = load_data(args.data, args.batch_size, 1)
    model = TokenAE(args.input_len, args.codebook_dim, args.embedding_dim, args.n_embeddings, args.kernel_size, args.vocab_size, args.alpha, args.beta, args.gamma).to(device)
    # optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, amsgrad=True, weight_decay=args.weight_decay)
    
    # Updated optimizer that provides different learning rates for the VQ and the model
    base_params = [p for name, p in model.named_parameters() if not name.startswith("vector_quantization")]
    optimizer = optim.Adam([
    {'params': base_params, 'lr': args.learning_rate, 'amsgrad':True, 'weight_decay':args.weight_decay}, 
    {'params': model.vector_quantization.parameters(), 'lr': args.learning_rate * 10, 'amsgrad':True, 'weight_decay':args.weight_decay}  
    ])

    # Actually do the learning rate scheduling
    warmup_steps = 1000  # Number of warmup steps
    total_steps = len(training_loader)  # Total training steps
    warmup_scheduler = LinearLR(optimizer, start_factor=0.1, total_iters=warmup_steps)
    cosine_scheduler = CosineAnnealingLR(optimizer, T_max=(total_steps - warmup_steps))
    scheduler = SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_steps])
    # scheduler = CosWithWarmup(optimizer, warmup_steps=1000, max_steps=len(training_loader), alpha_f=0.1)

    # Results to handle    
    results = {
        'n_updates': 0,
        'total loss': [],
        'reconstruction loss': [],
        'codebook loss': [],
        'commitment loss': [],
        'compression loss': [],
        'mask loss': [],
        'G Under Half': [],
        '% Codebook Used': [],
        '% Correct Masked Character': [],
        '% Correct Masked Token': [],
        '% Correct Predicted Character': [],
        '% Correct Predicted Token': [],
        'Compression': [],
        'LR': []
    }

    val_results = {
        "n_updates": 0,
        "val loss": [],
        "val recon loss": [],
        "val mask loss": [],
        "val codebook used": [],
        "val correct predicted char": [],
        "val correct predicted token": [],
        "val compression": []
    }

    # for epoch in range(args.epochs):
    #     print("\n------ EPOCH ", epoch, "------\n")
    model.train()
    train(optimizer, scheduler, training_loader, validation_loader, model, results, val_results)
