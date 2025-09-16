#!/bin/bash
#SBATCH --job-name=token_sweeps
#SBATCH --account=kempner_sham_lab
#SBATCH --partition=kempner
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=0-1:00
#SBATCH --mem=256G
#SBATCH --output=logfiles/test-tokenizer-g-reg_token-%A-%a.out
#SBATCH --error=logfiles/test-tokenizer-g-reg_token-%A-%a.err
#SBATCH --mail-type=END
#SBATCH --mail-user=theodatta@college.harvard.edu

# Load modules
module load python
# source /n/sw/Mambaforge-23.11.0-0/bin/activate

# Activate conda environment (optional)
echo "Activating conda environment..."
# source /n/sw/Mambaforge-23.11.0-0/bin/activate token_olmo2
# mamba activate token_olmo #if you have issues, it might be because we have different conda envs (token_olmo and token-olmo)
module load cuda/12.4.1-fasrc01
# mamba install numpy==1.26.4 --yes

# mamba run -n tokenae python /n/home03/tdatta/tank-vae/pipeline.py
# mamba run -n tokenae python /n/home03/tdatta/tank-vae/main.py --alpha 2.0 --vocab_size 2000 --codebook_dim 1024 --embedding_dim 1024 --use_wandb true --seed 1 --hardset 8 --name "h8_sweep2" 
# mamba run -n tokenae python /n/home03/tdatta/tank-vae/main.py --alpha 3.0 --vocab_size 5000 --codebook_dim 1440 --embedding_dim 1440 --use_wandb true --seed 1 --name "dim_sweep_long_1440" --batch_size 512 
# mamba run -n tokenae python /n/home03/tdatta/tank-vae/main.py --alpha 3.0 --vocab_size 5000 --codebook_dim 1024 --embedding_dim 1024 --use_wandb true --seed 1 --name "dim_sweep_long_1024" 
# mamba run -n tokenae python /n/home03/tdatta/tank-vae/main.py --alpha 3.0 --vocab_size 5000 --codebook_dim 720 --embedding_dim 720 --use_wandb true --seed 1 --name "dim_sweep_long_720" 
# mamba run -n tokenae python /n/home03/tdatta/tank-vae/main.py --alpha 3.0 --vocab_size 5000 --codebook_dim 512 --embedding_dim 512 --use_wandb true --seed 1 --name "dim_sweep_long_512" 

mamba run -n tokenae python /n/home03/tdatta/tank-vae/data/data_distribution.py

# mamba run -n tokenae python /n/home03/tdatta/tank-vae/regex_tokenizer.py --val true # --val true #--start 0.0 
# [0,0.0625,0.125,0.1875,0.25,0.3125,0.375,0.4375,0.5,0.5625,0.625,0.6875,0.75,0.8125,0.875,0.9375]
# [0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875]
# mamba run -n token_olmo2 python /n/home03/tdatta/tank-vae/pipeline.py --path bpe_tokenizer-2k.json --destination /n/holyscratch01/sham_lab/tokenae/2k_vocab/tinystories_bpe/data.npy


# source /n/sw/Mambaforge-23.11.0-0/bin/activate token_olmo2
# mamba deactivate
# source /n/sw/Miniforge3-24.7.1-0/bin/activate token_olmo2
# echo "Activated 'token_olmo2' environment."

# # Verify which Python is being used
# echo "Testing which Python is active:"
# which python
# python --version

# # Check if numpy is installed
# echo "Checking if numpy is installed and usable:"
# python -c "import numpy; print('Numpy version:', numpy.__version__)" || echo "Numpy is not installed or cannot be imported."

# # Check if other dependencies are installed (e.g., PyTorch)
# echo "Checking if PyTorch is installed and usable:"
# python -c "import torch; print('PyTorch version:', torch.__version__)" || echo "PyTorch is not installed or cannot be imported."


# #I have been having trouble with external imports
# # pip install x_transformers
# # pip install vector_quantize_pytorch
# # pip install git+https://github.com/DeMoriarty/fast_pytorch_kmeans.git

# #run the code
# echo "running the code"
# python /n/home03/tdatta/tank-vae/pipeline.py
