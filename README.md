<div align="center"><img src="assets/dromedario.alpha.png" width="200"></div>


<div align="center">

<a href="https://clic2026.unipa.it/" style="line-height: 0;"><img src="http://img.shields.io/badge/CLiC--it-2026-forestgreen.svg" style="display: block; margin: 0;"/></a>
<a href="https://clic2026.unipa.it/accepted-papers/" style="line-height: 0;"><img src="http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg" style="display: block; margin: 0;"/></a>
<a href="https://huggingface.co/datasets/sapienzanlp/dromedario-3-sft-dataset" style="line-height: 0;" target="_blank"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-FCD21D" style="display: block; margin: 0;"/></a>
<a href="https://opendatacommons.org/licenses/by/1-0/" style="line-height: 0;" target="_blank"><img src="https://img.shields.io/badge/License-ODC%20BY%201.0-green.svg" style="display: block; margin: 0;"/></a>

</div>


# 🐪 Dromedario 3

**Dromedario 3** is an Italian instruction-tuning dataset based on the English Tülu 3 SFT mixture. Instructions and responses are classified using the [Super-NaturalInstructions](https://github.com/allenai/natural-instructions) taxonomy, and the resulting classes are manually reviewed to determine which examples can be safely translated. Examples from the selected classes are then machine-translated into Italian.

The dataset can be used for supervised fine-tuning of Italian and Italian-English bilingual language models. It includes dialogue, reasoning, coding, and knowledge-intensive tasks, with both the original English data and the corresponding Italian translations.

## 🎯 TL;DR:

Dromedario 3 is freely available on [HuggingFace](https://huggingface.co/datasets/sapienzanlp/Dromedario_3) and can be downloaded with the `datasets` library.

```python
from datasets import load_dataset

ds = load_dataset("sapienzanlp/dromedario-3-sft-dataset", split="train")

# Italian subset: rows with a usable translation
it_only = ds.filter(lambda x: x["messages_transl"] is not None)

# Optionally drop rows that were automatically corrected
it_safe = it_only.filter(lambda x: not x["flags"])
```

Every row carries both the original conversation and, where available, its Italian translation, so English-only, Italian-only, and mixed training sets can all be built from this single release.

---

# 🚀 SFT training on CINECA
If you wish to train your model on Dromedario, you can follow these instructions.
The original experiments were carried out on the [Leonardo HPC](https://docs.hpc.cineca.it/hpc/leonardo.html) (CINECA); the provided scripts can be adapted to other SLURM-based clusters.
We performed all training runs using 4 nodes in parallel, each equipped with 4 GPUs with 64 GB of VRAM. 
We handled the parallelism with [DeepSpeed](https://www.deepspeed.ai/tutorials/zero/).

This repo itself must be cloned directly into `$SCRATCH_ROOT/dromedario` (i.e. what the scripts call
`$PROJECT_DIR` — see [Configuration](#configuration)): the `.sbatch` scripts `cd` there and expect
`./config/...` and `./_configs/...` to resolve relative to that directory.

This repo relies on two separate Python environments. The repo's own utility scripts
(`prepare_dataset.py`, `compose_yaml.py`, `download_models.py`) run under the `uv`-managed
environment pinned by `uv.lock` (Python 3.12, per `.python-version`/`pyproject.toml`) — run
`uv sync --locked` once, then invoke them with `uv run <script>.py` as shown below. Training instead
uses a separate Python 3.11 virtualenv (`llama_env`, created below) containing LLaMA-Factory, since
that is the Python version its dependencies target.

## 🛠️ Environment Setup

```bash
cd $SCRATCH_ROOT/dromedario   # clone/run everything from here on

# Sync the utility environment (prepare_dataset.py, compose_yaml.py, download_models.py)
uv sync --locked

# Create a separate virtual env for LLaMA Factory (training)
uv venv llama_env --python 3.11
source llama_env/bin/activate

# Clone LLaMA Factory
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# Use uv to sync dependencies
uv pip install -e .
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 --reinstall # check your cuda version beforehand
uv pip install "transformers==4.57.3" # Note: we need to pin transformers to deal with Minerva
uv pip install -r requirements/metrics.txt
uv pip install wandb 

pip install deepspeed-kernels deepspeed==0.14.4

# Log into a compute node and run this to check
srun -N 1 -A YOUR_PROJECT --ntasks-per-node=1 --cpus-per-task=8 --partition=boost_usr_prod --gres=gpu:4 --gpus-per-task=4 --time 00:15:00 --pty /bin/bash

module load cuda
python - <<'PY'
import deepspeed
from deepspeed.ops.op_builder import CPUAdamBuilder
print("deepspeed:", deepspeed.__version__)
print("cpu_adam:", CPUAdamBuilder().is_compatible())
try:
    from deepspeed.ops.op_builder import FusedAdamBuilder
    print("fused_adam:", FusedAdamBuilder().is_compatible())
except Exception as e:
    print("fused_adam import failed:", e)
PY

# If fused_adam is compatible, then all good. 
# Otherwise, you need to install deepspeed on a node with CUDA toolkit loaded.

cd "$SCRATCH_ROOT/dromedario"   # back to the repo root for the steps below
```


## 📊 Logging inside the compute node

You may want to log into a compute node for debugging.
To do so, set up a public key on the login node.

```bash
# Create a local SSH keypair on the cluster
ssh-keygen -t rsa -b 4096 -f ~/.ssh/id_rsa -N ""

# Append this newly created public key to your own authorized_keys file
cat ~/.ssh/id_rsa.pub >> ~/.ssh/authorized_keys

# Ensure strict, secure permissions (SSH will silently ignore keys if permissions are too wide)
chmod 700 ~/.ssh
chmod 600 ~/.ssh/authorized_keys
```

Then, you can connect to the compute node (only when the job is running) with:

```bash
ssh lrdnXXXX # use squeue -u $USER to see your assigned compute nodes
```


## ⚙️ Configuration

This repo is set up around our own CINECA Leonardo account; before running anything, edit the
following:

**Cluster paths** — every script derives its paths from a single `SCRATCH_ROOT` environment
variable (default: `/leonardo_scratch/large/userexternal/$USER`, the Leonardo scratch convention).
Export a different `SCRATCH_ROOT` before running any script if you are on another cluster/filesystem.
This repo itself must be cloned directly into `$SCRATCH_ROOT/dromedario` (called `$PROJECT_DIR` in
the scripts) — the `.sbatch` scripts `cd` there and resolve `./config/...` and `./_configs/...`
relative to it.

**SLURM account & partition** — `scripts/submit_all_configs.sh` requires `SLURM_ACCOUNT` to be set
in your environment (it has no usable default) and optionally `SLURM_PARTITION` (defaults to
`boost_usr_prod`, the Leonardo GPU partition); both are passed to `sbatch` and override whatever is
in the `.sbatch` files. If you instead sbatch `scripts/multinode_sft.sbatch` or
`scripts/singlenode_sft.sbatch` directly, note that SLURM does **not** expand shell variables inside
`#SBATCH` directives, so you must edit the `-A` (account) and `-p` (partition) lines in those files
by hand, or override them on the command line (`sbatch --account=... --partition=... ...`).

**HuggingFace token** — set `hf_hub_token` in [`config/dromedario_sft.yaml`](config/dromedario_sft.yaml)
(currently a placeholder, `hf_XXXX...`) if the models/dataset you use are gated.

**Dataset location** — pick one directory for the prepared dataset, pass it as `--output_dir` to
`prepare_dataset.py`, and set `dataset_dir` in every file under [`config/data/`](config/data) to that
same directory (currently a `path/to/dataset` placeholder in both places). Then copy
`config/dataset_info.json` into that directory as well — its keys already match the `dataset:`
fields in `config/data/*.yaml`, and this is also what `compose_yaml.py`'s output reminds you to do.

**Output location** — `output_dir` in [`config/dromedario_sft.yaml`](config/dromedario_sft.yaml)
defaults to `./saves/`; change it if you want checkpoints written elsewhere.

## ⚡ Training 

### Data

This script will download and convert the dataset into the shareGPT format. 
It will write all the recipes used in our paper to disk.
- `control`: Tülu 3 minus the 24K persona-based (`tulu_hard_coded`) instances
- `sub_50`/`sub_100`: substitute half/all translatable items with their Italian translation
- `add_50/add_100`: add Italian translations of half/all translatable items 

```bash
uv run prepare_dataset.py --output_dir path/to/dataset   # writes the ShareGPT jsonl files there
cp config/dataset_info.json path/to/dataset/dataset_info.json
```

`path/to/dataset` must be the same directory you set as `dataset_dir` in `config/data/*.yaml`
(see [Configuration](#configuration)).

### Models

Leonardo compute nodes do not have access to internet, so models should be downloaded before starting any run. 
You can do so by running:

```bash
./scripts/download_models.sh    # write the HF model id in the script 
```

### Training run

We ran all trainings through the [LLaMA-Factory library](https://llamafactory.readthedocs.io/en/latest/).
Refer to the official documentation available [here](https://llamafactory.readthedocs.io/en/latest/getting_started/sft.html) for the setup.

Before submitting the jobs, generate the per-run LLaMA-Factory configs by combining the base config
([`config/dromedario_sft.yaml`](config/dromedario_sft.yaml)) with each model config in
[`config/models/`](config/models) and each data config in [`config/data/`](config/data):

```bash
uv run compose_yaml.py     # writes one merged config per (model, dataset) pair into ./_configs/
```

This prints, for each generated config, the exact `llamafactory-cli train` command and the per-GPU
batch size, and reminds you to copy `config/dataset_info.json` into every `dataset_dir` referenced by
the data configs (see [Configuration](#configuration) below).

Remember to change the log directories in the `.sbatch` scripts accordingly.

```bash
./scripts/submit_all_configs.sh     # sbatches all configs found in ./_configs/
```
