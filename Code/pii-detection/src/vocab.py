from collections import defaultdict
from tqdm import tqdm

class Vocab:
    def __init__(self, tokens, min_freq=5, specials= None):
        if specials is None:
            specials = ["<pad>", "<unk>", "<bos>", "<eos>"]

        self.token_freq = defaultdict(int)
        self.min_freq = min_freq
        self.specials = specials
        self.itos = list(self.specials)
        self.stoi = {tok: i for i, tok in enumerate(specials)}

        if tokens is not None:
            self.build_vocab(tokens)

    def build_vocab(self, tokens):

        print("Building vocab...")
        for tok in tqdm(tokens):
            self.token_freq[tok] += 1

        for tok, freq in tqdm(self.token_freq.items()):
            if freq > self.min_freq:
                self.stoi[tok] = len(self.itos)
                self.itos.append(tok)
    def __len__(self):
        return len(self.itos)

    def token_to_id(self, token):
        if token in self.stoi:
            return self.stoi[token]

        return "<unk>"

    def id_to_token(self, id):

        if id < len(self.itos):
            return self.itos[id]

        return -1

    def encode(self, token):
        return self.token_to_id(token)

    def decode(self, id):
        return self.id_to_token(id)

    def save(self, path= "../data/vocab.txt"):
        with open(path, "w", encoding="utf-8") as f:
            for tok in self.itos:
                f.write(tok + "\n")

    @staticmethod
    def load(path="../data/vocab.txt"):
        with open(path, "r", encoding="utf-8") as f:
            tokens = [line.strip() for line in f.readlines()]
        vocab = Vocab(tokens=[], specials=[])
        vocab.itos = tokens
        vocab.stoi = {tok: i for i, tok in enumerate(tokens)}

        return vocab

if __name__ == "__main__":
    import os, json
    data_path = "../data/combined_ner_dataset.json"
    with open(data_path) as f:
        data = json.load(f)

    tokens = []
    for i in tqdm(data):
        for token in i["tokens"]:
            tokens.append(token)

    vocab = Vocab(tokens=tokens)
    vocab.build_vocab(tokens)
    vocab.save()

