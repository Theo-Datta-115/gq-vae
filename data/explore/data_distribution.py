"""
This evaluates the distribution of token usage frequencies for tokenized data
It outputs a histogram with overlapping bars to show the difference between distributions
Requirements: path to folder containing tokenized data in .npy files (one or many)
"""

import numpy as np
import os
import matplotlib.pyplot as plt

# Input the path to the folder you stored the tokenized data in (.npy file)
data = {
    "NAME1": "PATH1",
    "NAME2": "PATH2"
}

def main():
    for k, v in data.items():
        # read all files in the directory
        data_store = np.array([])
        for f in os.listdir(v):
            if f.endswith(".npy"):
                a = np.memmap(f"{v}/{f}", dtype=np.uint16, mode="r")
                data_store = np.concatenate((data_store, a))
        values, counts = np.unique(data_store, return_counts=True)
        
        counts = np.sort(counts)[::-1]
        counts = counts / np.sum(counts)

        plt.bar(np.arange(50), counts[:50], width=1, label=k)

    #plot
    plt.xlabel("Token")
    plt.ylabel("Cumulative Count Frequency")
    plt.title("Token Counts")
    plt.grid(True)

    plt.legend()
    plt.savefig("/n/home03/tdatta/tank-vae/data/hist2.png")

if __name__ == "__main__":
    main()

