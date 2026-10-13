"""Held-out template-grouped test report; no manual placeholders."""
from pathlib import Path
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from src.dataset import IntentDataset, collate_batch
from src.model import IntentClassifier

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / 'models' / 'classifier.pt'
REPORT_PATH = ROOT / 'reports' / 'evaluation_report.md'


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError('Train the model first: python -m src.train')
    ckpt = torch.load(MODEL_PATH, map_location='cpu', weights_only=True)
    cfg = ckpt['config']
    vocab, i2l = ckpt['vocab'], ckpt['label_map']
    l2i = {v: k for k, v in i2l.items()}
    model = IntentClassifier(cfg['vocab_size'], cfg['embed_dim'], cfg['hidden_dim'],
                             cfg['num_classes'], rnn_type=cfg['rnn_type'])
    model.load_state_dict(ckpt['model_state'])
    model.eval()
    ds = IntentDataset(ROOT / 'data' / 'test.csv', vocab, l2i, max_len=cfg['max_len'])
    loader = DataLoader(ds, batch_size=64, collate_fn=collate_batch)
    actual, predicted = [], []
    with torch.inference_mode():
        for ids, lengths, y in loader:
            actual.extend(y.tolist())
            predicted.extend(model(ids, lengths).argmax(-1).tolist())
    indices = list(range(len(i2l)))
    names = [i2l[i] for i in indices]
    acc = accuracy_score(actual, predicted)
    macro = f1_score(actual, predicted, average='macro', zero_division=0)
    detail = classification_report(actual, predicted, labels=indices, target_names=names, digits=3, zero_division=0)
    cm = confusion_matrix(actual, predicted, labels=indices)
    lines = ['# Nexa · Intent Classifier Evaluation', '',
             f'- Test examples: **{len(ds)}** (held-out template families)',
             f'- Test accuracy: **{acc:.4f}**', f'- Test macro-F1: **{macro:.4f}**',
             '- Caveat: entirely synthetic data; production accuracy is not established.',
             '', '## Per-class report', '', '```text', detail.rstrip(), '```', '',
             '## Confusion matrix', '', '| True / Predicted | ' + ' | '.join(names) + ' |',
             '| :--- | ' + ' | '.join(['---:'] * len(names)) + ' |']
    for name, row in zip(names, cm):
        lines.append('| ' + name + ' | ' + ' | '.join(map(str, row)) + ' |')
    errors = [(names[i], names[j], int(cm[i][j]))
              for i in indices for j in indices if i != j and cm[i][j] > 0]
    lines.extend(['', '## Off-diagonal confusions', ''])
    lines.extend([f'- {true} → {pred}: {n} test messages; wrong specialist could give a less useful answer.'
                  for true, pred, n in sorted(errors, key=lambda x: -x[2])]
                 or ['- No off-diagonal mistakes on this synthetic holdout.'])
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Test accuracy={acc:.4f}; macro-F1={macro:.4f}; wrote {REPORT_PATH}')
    print(detail)

if __name__ == '__main__':
    main()
