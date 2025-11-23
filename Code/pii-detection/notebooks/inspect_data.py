#%%
import json
import numpy as np
import pandas as pd

#%%
import os
print(os.getcwd())
print(os.listdir("./Code"))
os.chdir("./Code/pii-detection/")

#%%
with open("data/combined_ner_dataset.json") as f:
    data = json.load(f)
print(len(data))
#%%
from collections import defaultdict
labels = defaultdict(int)
for i in data:
    for lbl in i["labels"]:
        labels[lbl] += 1
print(labels)
print(len(labels))
print(sorted(labels.items(), key=lambda x: x[1], reverse=True))
#%%
vocab = defaultdict(int)
for i in data:
    for wrd in i["tokens"]:
        vocab[wrd.lower().strip()] += 1
print(len(vocab))
# print(sorted(vocab.items(), key=lambda x: x[1], reverse=True))
#%%
print(len([wrd for wrd in vocab if vocab[wrd] > 10]))
