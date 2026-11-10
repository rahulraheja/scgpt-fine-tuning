# scGPT Perturbation Prediction

Fine-tune [scGPT](https://github.com/bowang-lab/scGPT) on Perturb-seq data to predict how gene knockouts change expression. Compares against simple baselines.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate && pip install --upgrade pip

# Install in order (torch first, then PyG, then the rest)
pip install torch==2.2.2 torchtext==0.17.2
pip install torch_geometric==2.5.3
pip install -r requirements.txt

# scGPT from source (editable, so you can explore internals)
git clone https://github.com/bowang-lab/scGPT.git ../scgpt-repo
pip install -e ../scgpt-repo --no-deps

# Register Jupyter kernel
python -m ipykernel install --user --name scgpt-fine-tuning --display-name "scGPT Fine-tuning"
```

Download the pretrained checkpoint (~205MB):

```python
from huggingface_hub import hf_hub_download
import os
os.makedirs("pretrained/scGPT_human", exist_ok=True)
for f in ["best_model.pt", "args.json", "vocab.json"]:
    hf_hub_download("u-cellm/scGPT-whole-human", filename=f, local_dir="pretrained/scGPT_human")
```

## Project Structure

```
notebooks/
  01_data_exploration.ipynb   # load Adamson dataset, explore structure, validate split
  02_baseline.ipynb           # MeanControl and DeltaMean baselines with metrics
  03_scgpt_finetune.ipynb     # fine-tune scGPT, checkpoint saved to outputs/runs/
  04_evaluation.ipynb         # compare scGPT vs baselines, generate plots
src/
  config.py                   # all hyperparams; mac/gpu switches live here
  model_utils.py              # vocab loading, model build, pretrained weight loading
  baselines.py                # MeanControlBaseline, DeltaMeanBaseline
  train.py                    # training loop + eval_perturb
  evaluate.py                 # compute_metrics wrapping scgpt.utils
```

Data and pretrained checkpoints are gitignored. The Adamson dataset (~141MB) downloads automatically via GEARS on first run of notebook 01.

## Mac / CPU Notes

MPS (Apple Silicon GPU) cannot handle the large attention buffers needed for this model. The config defaults to CPU with `max_seq_len=512`. Expect ~30–60 min/epoch on Adamson.

When switching to an NVIDIA GPU, change in `src/config.py`:
```python
force_cpu = False
use_fast_transformer = True   # requires: pip install flash-attn
amp = True
max_seq_len = 1536
```

## Datasets

| Dataset | Cells | Perturbations | Download |
|---------|-------|---------------|---------|
| Adamson 2016 (start here) | ~68K | 87 CRISPRi | auto via GEARS |
| Replogle 2022 k562_essential (scale-up) | ~400K | 2057 CRISPRi | auto via GEARS |

Change dataset: `pert_data.load(data_name="replogle_k562_essential")`

## Key Metrics

| Metric | What it measures | Note |
|--------|-----------------|------|
| `pearson` | Absolute expression, all genes | Misleadingly high for all models (>0.98) |
| `pearson_delta` | Delta from ctrl, all genes | More informative |
| `pearson_de` | Absolute expression, top-20 DE genes | Also high for baselines |
| **`pearson_de_delta`** | **Delta from ctrl, top-20 DE genes** | **The signal metric** |

Expected on Adamson test set:
- scGPT: `pearson_de_delta` ~0.80
- DeltaMean baseline: `pearson_de_delta` ~0.73
- MeanControl baseline: `pearson_de_delta` ~0.00

## Gotchas

- `pert_data.split` is a string (`"sim"`), not a dict. Use `pert_data.set2conditions` for `{"train": [...], "val": [...], "test": [...]}`.
- Conditions use `"GENE+ctrl"` format (e.g., `"CREB1+ctrl"`).
- `scvi-tools` is incompatible with PyTorch 2.x — it's not needed and is excluded.
- `scgpt==0.2.5` is not on PyPI; install from source.
- Metric `pearson_de` is high for all models including MeanControl — focus on `pearson_de_delta`.
