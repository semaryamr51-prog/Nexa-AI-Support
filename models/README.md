# Bundled intent checkpoint

`classifier.pt` is **included** in this ZIP and is loaded by `src/classify.py`. It was trained with the PyTorch bidirectional LSTM using the unique, source-template-grouped splits under `data/`.

Run `python -m src.evaluate` to reproduce the held-out model-only evaluation in `reports/evaluation_report.md`. The `src.classify.classify` production router adds a cautious rules layer for clear category terms and a cold-start fallback for environments without PyTorch or this checkpoint; the model-only report is **not** a measurement of this hybrid router.

The dataset is synthetic and not a guarantee of real-user accuracy. Run `python -m src.train` to retrain, or `python -m src.train --rnn_type gru --out models/classifier_gru.pt` for a separate GRU experiment. Avoid overwriting a validated checkpoint unintentionally.
