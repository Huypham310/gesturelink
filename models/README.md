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

Written by the training phase (`models/*.pt`), also gitignored.
