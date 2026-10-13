"""Deterministic, early-stopped classifier training (PyTorch LSTM/GRU)."""
from __future__ import annotations
import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader
from src.dataset import (IntentDataset, build_vocab, build_label_map, read_csv,
                         collate_batch, MAX_LEN)
from src.model import IntentClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / 'data'
MODELS = ROOT / 'models'


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval()
    y_true, y_pred = [], []
    with torch.inference_mode():
        for ids, lengths, labels in loader:
            logits = model(ids.to(device), lengths.to(device))
            y_true.extend(labels.tolist())
            y_pred.extend(logits.argmax(-1).cpu().tolist())
    return float(accuracy_score(y_true, y_pred)), float(f1_score(y_true, y_pred, average='macro', zero_division=0))


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--rnn_type', choices=['lstm', 'gru'], default='lstm')
    parser.add_argument('--embed_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=18)
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default=str(MODELS / 'classifier.pt'))
    args = parser.parse_args(argv)
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    texts, labels = read_csv(DATA / 'train.csv')
    vocab = build_vocab(texts)
    l2i, i2l = build_label_map(labels)
    train = IntentDataset(DATA / 'train.csv', vocab, l2i)
    val = IntentDataset(DATA / 'val.csv', vocab, l2i)
    train_loader = DataLoader(train, batch_size=args.batch_size, shuffle=True, collate_fn=collate_batch)
    val_loader = DataLoader(val, batch_size=args.batch_size, collate_fn=collate_batch)
    model = IntentClassifier(len(vocab), args.embed_dim, args.hidden_dim, len(l2i), rnn_type=args.rnn_type).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    best_f1, no_improve, history = -1.0, 0, []
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for ids, lengths, y in train_loader:
            optimizer.zero_grad()
            logits = model(ids.to(device), lengths.to(device))
            loss = criterion(logits, y.to(device))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += loss.item() * len(y)
        accuracy, f1 = evaluate(model, val_loader, device)
        history.append({'epoch': epoch, 'train_loss': round(running/len(train), 4),
                        'val_accuracy': round(accuracy, 4), 'val_macro_f1': round(f1, 4)})
        print(f"Epoch {epoch:02d} loss={running/len(train):.3f} val_acc={accuracy:.3f} val_macro_f1={f1:.3f}", flush=True)
        if f1 > best_f1 + 1e-4:
            best_f1, no_improve = f1, 0
            torch.save({'model_state': model.state_dict(), 'vocab': vocab, 'label_map': i2l,
                        'config': {'vocab_size': len(vocab), 'embed_dim': args.embed_dim,
                                   'hidden_dim': args.hidden_dim, 'num_classes': len(l2i),
                                   'rnn_type': args.rnn_type, 'max_len': MAX_LEN}}, dest)
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print('Early stopping', flush=True)
            break
    path = ROOT / 'reports' / 'training_history.json'
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({'seed': args.seed, 'split': 'template-grouped',
                                'best_validation_macro_f1': best_f1,
                                'epochs': history}, indent=2), encoding='utf-8')
    print(f'Saved checkpoint: {dest}; metrics: {path}')

if __name__ == '__main__':
    main()
