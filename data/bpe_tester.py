from tokenizers import Tokenizer
import torch
from datasets import load_dataset

ds = load_dataset("roneneldan/TinyStories", split='validation[:2%]').with_format("torch")
text_column = ds['text']

unicode_values = [ord(char) for string in text_column for char in string]

data = torch.tensor(unicode_values, dtype=torch.int32)
data = data[data < 128]
original_text_size = len(data)

def compute_compression(tokenizer_file_path):
    print(f"------- computing compression for {tokenizer_file_path}")
    tokenizer = Tokenizer.from_file(tokenizer_file_path)

    tokenized_texts = [tokenizer.encode(text) for text in text_column]
    tokenized_text_size = sum(len(tokens) for tokens in tokenized_texts)

    compression_rate = original_text_size / tokenized_text_size

    print(f"Original text size (number of characters): {original_text_size}")
    print(f"Tokenized text size (number of tokens): {tokenized_text_size}")
    print(f"Compression rate: {compression_rate}")
    
    return original_text_size, tokenized_text_size, compression_rate



import pandas as pd
compression_rates = {'vocab_size': [], 'compression_rate': [], 'tokens': []}
vocabs = [2000, 5000, 10000, 20000, 40000] 

for vocab in vocabs:
    _, tokens, compression = compute_compression(f"./tokenizer-ts-{vocab}.json")
    compression_rates['vocab_size'].append(vocab)
    compression_rates['compression_rate'].append(compression)
    compression_rates['tokens'].append(tokens)

df = pd.DataFrame(compression_rates)
df.to_csv('compression_rates.csv', index=False)
print("Compression rates saved to compression_rates.csv")
