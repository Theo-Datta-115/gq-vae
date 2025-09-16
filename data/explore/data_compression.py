"""
This evaluates the compression (char/token) of tokenized data
Requirements: path to folder containing tokenized data in .npy files (one or many)
"""

import numpy as np
import os

# Input the path to the folder you stored the tokenized data in (.npy file)
data = {
    "NAME1": "PATH1",
    "NAME2": "PATH2"
}

# Found by looking at the length of the val and train sets for tinystories
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