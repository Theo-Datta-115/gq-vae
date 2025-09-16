import numpy as np
import os
import matplotlib.pyplot as plt

data = {
    # "tiny-gq-10k": "/n/holyscratch01/sham_lab/tokenae/retry/tinystories_tokenized/",
    "BPE": "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/bpe10000/val/",
    "GQ-VAE": "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k_v2/train/",
    # "tiny-gq-10k-val2": "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k/train/",
    # "tiny-bpe-10k": "/n/holyscratch01/sham_lab/tokenae/tinystories_bpe"
}
val_chars = 19190318
train_chars = 189997320


def main():
    for k, v in data.items():
        # read all files in the directory
        data_store = np.array([])
        for f in os.listdir(v):
            if f.endswith(".npy"):
                a = np.memmap(f"{v}/{f}", dtype=np.uint16, mode="r")
                data_store = np.concatenate((data_store, a))
                # print(len(data_store))
        values, counts = np.unique(data_store, return_counts=True)
        
        print(counts)
        #sort the counts
        counts = np.sort(counts)[::-1]

        #save the values and the counts to a file
        # np.save(f"{k}_values.npy", values)
        # np.save(f"{k}_counts.npy", counts)

        #normalize the counts
        counts = counts / np.sum(counts)

        # plt.bar(np.arange(len(counts)), counts, width=1)
        plt.bar(np.arange(50), counts[:50], width=1, label=k)
        # counts = np.cumsum(counts)
        # plt.plot(counts)    

    #add labels
    plt.xlabel("Token")
    plt.ylabel("Cumulative Count Frequency")
    plt.title("Token Counts")
    plt.grid(True)

    #add a legend
    plt.legend()

    #save the plot  
    plt.savefig("/n/home03/tdatta/tank-vae/data/hist2.png")
"""
import numpy as np
import os

data = {
    "tiny-gq-10k-retry": "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k_v2/train",
    "tiny-gq-10k-retry-val": "/n/netscratch/sham_lab/Everyone/tdatta/tokenae/retry/set_gates/tinystories_50k_v2/val",
    "tiny-bpe-10k": "/n/holyscratch01/sham_lab/tokenae/tinystories_bpe",
    "tiny-bpe-10k-val": "/n/holyscratch01/sham_lab/tokenae/tinystories_bpe_val",
}
val_chars = 19190318
train_chars = 1899973203


def main():
    for k, v in data.items():
        # print(k)
        # read all files in the directory
        length = 0
        for f in os.listdir(v):
            # print(f)
            if f.endswith(".npy"):
                a = np.memmap(f"{v}/{f}", dtype=np.uint16, mode="r")
                length += len(a)
        val = "val" in k
        if val:
            print(f"{k}: {length} tokens -> {val_chars / length:.4f}x")
        else:
            print(f"{k}: {length} tokens -> {train_chars / length:.4f}x")


if __name__ == "__main__":
    main()
"""

if __name__ == "__main__":
    main()

