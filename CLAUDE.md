# CLAUDE.md — AI Assistant Guide for normalizing-flows-tutorial

## Project Overview

This is an educational repository implementing **normalizing flows** — a class of generative models that learn invertible transformations to map between simple distributions (e.g., Gaussian) and complex, high-dimensional data distributions.

The tutorial is authored by [Eric Jang](http://blog.evjang.com) and accompanies two blog posts:
- [Part 1](http://blog.evjang.com/2018/01/nf1.html): Introduction to normalizing flows
- [Part 2](http://blog.evjang.com/2018/01/nf2.html): Modern normalizing flow architectures

**Important:** The notebooks use **TensorFlow 1.5** (specifically `tf.contrib` APIs, which were removed in TF 2.0). Do not attempt to upgrade them to TF 2.x without a full rewrite.

---

## Repository Structure

```
normalizing-flows-tutorial/
├── nf_part1_intro.ipynb      # Tutorial Part 1 — custom invertible 2D flows
├── nf_part2_modern.ipynb     # Tutorial Part 2 — Real-NVP, BatchNorm, MAF
├── siggraph.pkl              # Dataset: SIGGRAPH letters point cloud (229 KB)
├── salt_bae.png              # Image asset used in Part 2 notebook (475 KB)
├── launch_jupyter.sh         # Starts Jupyter server on port 8889
├── hello_world/
│   └── index.html            # Standalone HTML5 calculator app (no dependencies)
├── README.md                 # Minimal readme with blog links
└── LICENSE                   # MIT License (2018, Eric Jang)
```

---

## Key Concepts & Architecture

### Normalizing Flows

A normalizing flow transforms a base distribution (standard normal) through a chain of invertible, differentiable functions (bijectors). The transformed distribution can model complex data.

Core components implemented in the notebooks:

| Component | Location | Description |
|-----------|----------|-------------|
| `LeakyReLU` | Part 1 | Custom bijector with element-wise invertible activation |
| `AffineCouplingLayer` | Part 1 | Low-rank affine update for 2D flows |
| `NVPCoupling` | Part 2 | Real-NVP affine coupling with learned masks |
| `BatchNorm` | Part 2 | Batch normalization bijector with exponential moving average |
| `MAF/IAF` | Part 2 | Masked Autoregressive / Inverse Autoregressive Flows |

### Training Setup

- **Base distribution:** `MultivariateNormalDiag` (isotropic Gaussian)
- **Loss:** Negative log-likelihood of data samples
- **Optimizer:** Adam (default lr ≈ 1e-3)
- **Steps:** ~100,000 gradient steps (Part 1), varies by dataset (Part 2)
- **Datasets:** Two Moons (sklearn), Gaussian mixtures, SIGGRAPH letters

---

## Development Environment

### Dependencies (TF 1.x Era)

There is no `requirements.txt`. The notebooks require:

```
tensorflow==1.5.0  # Must use tf.contrib APIs (removed in TF 2.0)
numpy
matplotlib
scikit-learn
```

**Recommended approach:** Use a Python 3.6 virtual environment or a Docker image pinned to TensorFlow 1.5.

### Running the Notebooks

```bash
# Start Jupyter on port 8889 (no browser)
bash launch_jupyter.sh

# Or manually:
jupyter notebook --no-browser --port 8889
```

Then open `nf_part1_intro.ipynb` first, followed by `nf_part2_modern.ipynb`.

---

## Notebook Walkthrough

### Part 1 — `nf_part1_intro.ipynb`

1. Generates toy 2D target distributions (Gaussian mixture, nonlinear)
2. Constructs invertible flow using a chain of `LeakyReLU` + affine coupling bijectors
3. Trains via negative log-prob loss using Adam
4. Visualizes intermediate transformations across layers
5. Evaluates density coverage of learned distribution

### Part 2 — `nf_part2_modern.ipynb`

1. **Dataset selection:** Toggle between `siggraph`, `two_moons`, or `gaussian`
2. **Model config:** Select `real_nvp`, `maf`, or `iaf` at the top of the notebook
3. **Optional BatchNorm:** Toggle `use_batch_norm` flag
4. Implements permutation layers to shuffle dimensions between coupling layers
5. Trains on 2D point clouds and visualizes learned density
6. Includes a "bonus" section with `salt_bae.png` image

---

## Data Files

### `siggraph.pkl`
- Pickled Python object (NumPy array) containing a 2D point cloud forming SIGGRAPH conference letters
- Load with: `import pickle; data = pickle.load(open('siggraph.pkl', 'rb'))`
- Used as the target distribution for Part 2 training

---

## `hello_world/` Subdirectory

This directory contains a standalone mobile-friendly calculator web app (`index.html`) that is **unrelated to the normalizing flows tutorial**. It was added as a separate feature.

- Pure HTML/CSS/JavaScript — no build step, no dependencies
- Glass-morphism styling, responsive layout
- Supports keyboard input, division-by-zero error handling
- Open directly in a browser: `open hello_world/index.html`

---

## Conventions & Coding Style

- **TensorFlow 1.x style:** All code uses `tf.Session`, `tf.placeholder`, `tf.contrib.distributions`
- **No type annotations, no unit tests** — this is a tutorial repo, not production code
- **Self-contained cells:** Each notebook cell can be read independently; cells should be run top-to-bottom
- **In-cell visualization:** All plots are rendered inline using `matplotlib`
- **No package structure:** No `__init__.py`, no importable modules

---

## What NOT to Change

- **Do not upgrade to TensorFlow 2.x** — the entire codebase depends on `tf.contrib` APIs that no longer exist. Any upgrade requires a complete rewrite using TFP (TensorFlow Probability) or a different framework.
- **Do not add `requirements.txt` unless pinning to TF 1.5** — generic dependency files will cause version conflicts.
- **Do not refactor notebooks into `.py` modules** without explicit user request — notebooks are intentionally sequential and pedagogical.

---

## Common Tasks

### Add a New Bijector

Implement a class inheriting from `tf.contrib.distributions.bijectors.Bijector`:

```python
class MyBijector(tf.contrib.distributions.bijectors.Bijector):
    def __init__(self, ...):
        super().__init__(forward_min_event_ndims=0, name='my_bijector')

    def _forward(self, x):
        ...

    def _inverse(self, y):
        ...

    def _forward_log_det_jacobian(self, x):
        ...
```

### Add a New Dataset

In Part 2, datasets are selected by name. To add a new one, look for the `if dataset == '...'` block and add a new branch returning a NumPy array of shape `(N, 2)`.

### Modify the Calculator App

Edit `hello_world/index.html` directly — no build step required. The file is self-contained.

---

## Git Workflow

- Default branch: `master`
- Feature/AI branches use the pattern: `claude/<description>-<session-id>`
- Commit messages should be descriptive (e.g., `Add BatchNorm bijector to Part 2`)
- No CI/CD, no linting, no pre-commit hooks

---

## License

MIT License — Copyright 2018 Eric Jang. See `LICENSE` for full text.
