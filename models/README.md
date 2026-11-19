# Models

Model files are not committed (`*.task`, `*.pt`, `*.onnx` are gitignored).

## hand_landmarker.task (MediaPipe)

Free model from Google, used by the MediaPipe Tasks `HandLandmarker`. No account or API key.

```bash
gesturelink download-model
# -> models/hand_landmarker.task (about 7.5 MB)
```

Same thing by hand (PowerShell):

```powershell
Invoke-WebRequest `
  -Uri https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task `
  -OutFile models/hand_landmarker.task
```

or `curl -L -o models/hand_landmarker.task <same url>`.

Options: `--dest PATH` to save somewhere else, `--url URL` to use a mirror, `--force` to re-download.

The old `mp.solutions.hands` API is gone in mediapipe 1.x, so this project only uses
`mediapipe.tasks.python.vision.HandLandmarker`.

## Classifier weights

Written by `gesturelink train`:

```bash
gesturelink prepare                 # data/raw -> data/processed/dataset.npz
gesturelink train                   # -> models/gesturelink.pt + models/metrics.json
```

- `models/gesturelink.pt`: weights plus everything needed to rebuild the model (architecture,
  label names, the training config, val accuracy). Loaded with `torch.load(weights_only=True)`,
  so it only contains tensors and plain Python types. Gitignored.
- `models/metrics.json`: val accuracy, macro F1, per-class precision / recall / F1, confusion
  matrix, best epoch, training time and the per-epoch loss history. Gitignored too.

The network is a small MLP `63 -> 128 -> 64 -> 7` (ReLU, dropout 0.1, 16,903 parameters) trained
with AdamW and cross entropy. The best epoch by val accuracy is the one saved. Hyperparameters
are in `configs/train.json`; every key can also be overridden from the CLI
(`--epochs`, `--lr`, `--batch-size`, `--seed`, `--data`, `--out-dir`).

Sanity check without a camera (all numbers are on synthetic data, so they only show that the
pipeline works, not how well it does on a real hand):

| data | val accuracy |
| --- | --- |
| `tests/fixtures/landmarks/samples.csv` (112 train / 28 val) | 0.964 |
| `synthesize --per-class 150` (840 train / 210 val) | 1.000 |

Both train on CPU in under 3 seconds. Real recordings will be noisier, so expect lower numbers.
