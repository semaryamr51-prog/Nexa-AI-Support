# Nexa · Intent Classifier Evaluation

- Test examples: **115** (held-out template families)
- Test accuracy: **0.5478**
- Test macro-F1: **0.4252**
- Caveat: entirely synthetic data; production accuracy is not established.

## Per-class report

```text
              precision    recall  f1-score   support

     billing      0.588     0.682     0.632        44
     booking      0.267     0.182     0.216        22
     general      0.000     0.000     0.000        17
   technical      0.806     0.906     0.853        32

    accuracy                          0.548       115
   macro avg      0.415     0.442     0.425       115
weighted avg      0.500     0.548     0.520       115
```

## Confusion matrix

| True / Predicted | billing | booking | general | technical |
| :--- | ---: | ---: | ---: | ---: |
| billing | 30 | 4 | 10 | 0 |
| booking | 8 | 4 | 3 | 7 |
| general | 10 | 7 | 0 | 0 |
| technical | 3 | 0 | 0 | 29 |

## Off-diagonal confusions

- billing → general: 10 test messages; wrong specialist could give a less useful answer.
- general → billing: 10 test messages; wrong specialist could give a less useful answer.
- booking → billing: 8 test messages; wrong specialist could give a less useful answer.
- booking → technical: 7 test messages; wrong specialist could give a less useful answer.
- general → booking: 7 test messages; wrong specialist could give a less useful answer.
- billing → booking: 4 test messages; wrong specialist could give a less useful answer.
- booking → general: 3 test messages; wrong specialist could give a less useful answer.
- technical → billing: 3 test messages; wrong specialist could give a less useful answer.
