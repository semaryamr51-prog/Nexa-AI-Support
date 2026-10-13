"""
dataset.py
----------
Tokenization, vocabulary building, and a torch Dataset for the intent
classifier. Word-level tokenization is sufficient at this dataset size
(section 5.4 of the project brief).
"""

import csv
import re
from pathlib import Path
from collections import Counter

import torch
from torch.utils.data import Dataset

PAD_TOKEN, UNK_TOKEN = "<pad>", "<unk>"
MAX_LEN = 40
MAX_VOCAB = 10000


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r"[^a-z0-9'\s]", " ", text)
    return text.split()


def read_csv(path: Path):
    texts, labels = [], []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            texts.append(row["text"])
            labels.append(row["label"])
    return texts, labels


def build_vocab(train_texts, max_vocab=MAX_VOCAB):
    """Build vocab from the TRAINING split only, per the brief (5.4)."""
    counter = Counter()
    for t in train_texts:
        counter.update(tokenize(t))
    most_common = counter.most_common(max_vocab)
    vocab = {PAD_TOKEN: 0, UNK_TOKEN: 1}
    for tok, _ in most_common:
        if tok not in vocab:
            vocab[tok] = len(vocab)
    return vocab


def build_label_map(labels):
    unique = sorted(set(labels))
    label2id = {l: i for i, l in enumerate(unique)}
    id2label = {i: l for l, i in label2id.items()}
    return label2id, id2label


def encode(text: str, vocab: dict, max_len: int = MAX_LEN):
    ids = [vocab.get(tok, vocab[UNK_TOKEN]) for tok in tokenize(text)][:max_len]
    length = max(len(ids), 1)  # avoid zero-length sequences
    ids = ids + [vocab[PAD_TOKEN]] * (max_len - len(ids))
    return ids, length


class IntentDataset(Dataset):
    def __init__(self, csv_path, vocab, label2id, max_len=MAX_LEN):
        self.texts, self.labels = read_csv(csv_path)
        self.vocab = vocab
        self.label2id = label2id
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        ids, length = encode(self.texts[idx], self.vocab, self.max_len)
        label = self.label2id[self.labels[idx]]
        return (
            torch.tensor(ids, dtype=torch.long),
            torch.tensor(length, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
        )


def collate_batch(batch):
    ids = torch.stack([b[0] for b in batch])
    lengths = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return ids, lengths, labels
