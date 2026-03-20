# Liquid AI Vision Kit

A lightweight image classification toolkit built on **Liquid Neural Networks (LNNs)** — entirely in **NumPy**, no deep-learning framework required.

LNNs are inspired by the continuous-time dynamics of biological neurons. Instead of fixed activations, hidden states evolve according to an ordinary differential equation (ODE), giving the network an inherent temporal structure.

---

## Architecture

```
Image (H×W)
    │
    ▼
PatchEmbedder
  • Splits image into non-overlapping (patch_size × patch_size) patches
  • Flattens each patch to a 1-D vector
    │
    ▼  (iterate over patches)
LiquidVisionCell(s)
  • ODE: dh/dt = (−h + tanh(W_h h + W_in x + b)) / τ
  • Integrated with Euler steps
  • Hidden state carries temporal context across patches
    │
    ▼
Linear head  →  class logits
```

---

## Installation

```bash
git clone https://github.com/danieleschmidt/liquid-ai-vision-kit.git
cd liquid-ai-vision-kit
pip install numpy pytest
```

No other dependencies are required.

---

## Quick Start

```python
import numpy as np
from liquid_ai_vision_kit import LiquidVisionClassifier

# Build classifier
clf = LiquidVisionClassifier(
    image_size=16,
    patch_size=4,
    hidden_dim=64,
    n_classes=10,
)

# Forward pass on a random 16×16 image
img = np.random.rand(16, 16)
logits = clf.forward(img)
print(logits)          # shape (10,)
print(logits.argmax()) # predicted class
```

---

## Components

### `LiquidVisionCell`

ODE-based recurrent cell.

```python
from liquid_ai_vision_kit import LiquidVisionCell

cell = LiquidVisionCell(input_dim=16, hidden_dim=32, tau=1.0, dt=0.1)
x = np.random.randn(16)
output, hidden = cell.forward(x)          # first call, h initialised to zeros
output, hidden = cell.forward(x, h=hidden) # carry hidden state forward
```

| Parameter   | Description                                |
|-------------|--------------------------------------------|
| `input_dim` | Feature vector size                        |
| `hidden_dim`| Hidden state size                          |
| `tau`       | ODE time constant (larger = slower dynamics)|
| `dt`        | Euler step size                            |
| `n_steps`   | Integration steps per forward call         |

### `PatchEmbedder`

Splits a 2-D image into flattened patch vectors.

```python
from liquid_ai_vision_kit import PatchEmbedder

emb = PatchEmbedder(patch_size=4)
patches = emb.forward(img)  # shape: (n_patches, patch_dim)
```

### `LiquidVisionClassifier`

End-to-end pipeline.

```python
from liquid_ai_vision_kit import LiquidVisionClassifier

clf = LiquidVisionClassifier(
    image_size=32,
    patch_size=4,
    hidden_dim=64,
    n_classes=10,
    n_cells=2,    # stack two LiquidVisionCells
)
logits = clf.forward(img)
```

---

## Demo

Run the included demo to see inference on synthetic blob images and a latency benchmark:

```bash
python demo.py
```

---

## Tests

```bash
~/anaconda3/bin/python3 -m pytest tests/ -v
```

---

## Design Notes

- **Pure NumPy** — runs anywhere Python runs; no GPU required.
- **Euler integration** — simple, fast, and stable for small `dt`.
- **Patch-as-sequence** — treats image patches as a time series, letting the ODE cell build up spatial context recurrently.
- **Stacked cells** — `n_cells > 1` creates a deeper liquid hierarchy.

---

## License

MIT
