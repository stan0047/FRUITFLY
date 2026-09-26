# Notebooks

Exploratory notebooks only. Anything that a script or test depends on belongs
in `src/flybrain/` or `scripts/`, not here.

## Free compute workflow (Google Colab free tier)

Colab's free tier gives an NVIDIA T4 with a time limit. Keep sessions short and
push intermediate results to the repository (configs, small CSV summaries,
figures) rather than model checkpoints.

```python
# 1. Clone / mount the repo
!git clone <your-fork-url> flybrain
%cd flybrain

# 2. Free-tier GPU runtime: Runtime > Change runtime type > T4 GPU
!pip install -q -r requirements.txt
!pip install -q torch          # Colab build: picks up CUDA if a GPU is present

# 3. Same entry points as local
!python -m pytest -q
!python scripts/smoke_test.py
```

Check what you actually got, and never assume a GPU:

```python
import torch
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))
```

Guidelines:

* Request GPU only when a run is genuinely large. The scaffold runs on CPU.
* Pin a seed and save the resolved config next to any result.
* Colab's free tier is not reproducible infrastructure: for anything you intend
  to report, re-run locally or on a documented free service and record versions.
