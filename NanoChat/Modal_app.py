"""
modal_app.py

Pulls the already-tokenized shards from HF into a Modal Volume, then
runs Train.py on a GPU against them.
"""

import os
import modal

HF_DATASET_REPO = "seif-222/Tokenized_FineWebEdu_Shards"

APP_NAME = "nanochat-training"
VOLUME_NAME = "nanochat-data"

app = modal.App(APP_NAME)
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.5.1",
        "numpy",
        "tiktoken",
        "tqdm",
        "requests",
        "wandb",
        "huggingface_hub",
    )
    .add_local_file("Config.py", "/root/nanochat/Config.py")
    .add_local_file("Model.py", "/root/nanochat/Model.py")
    .add_local_file("DataLoader.py", "/root/nanochat/DataLoader.py")
    .add_local_file("Tokenizer.py", "/root/nanochat/Tokenizer.py")
    .add_local_file("optimizer.py", "/root/nanochat/optimizer.py")
    .add_local_file("HellaSwag.py", "/root/nanochat/HellaSwag.py")
    .add_local_file("Train.py", "/root/nanochat/Train.py")
)

MOUNT_PATH = "/data"
VOLUMES = {MOUNT_PATH: volume}


@app.function(
    image=image,
    volumes=VOLUMES,
    cpu=(4, 8),
    memory=(16384, 32768),
    timeout=60 * 60 * 3,
)
def download_data_remote():
    """Downloads the tokenized shards from the HF dataset repo into the Modal volume."""
    from huggingface_hub import snapshot_download
    try:
        snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            local_dir=f"{MOUNT_PATH}/fineweb-edu_tokenized",
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="A100-80GB",
    cpu=(4, 8),
    memory=(32768, 65536),
    timeout=60 * 60 * 14,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_model_remote():
    """Runs Train.py on a single GPU against the data in the volume."""
    import subprocess
    try:
        subprocess.run(
            ["python", "Train.py"],
            cwd="/root/nanochat",
            env={**os.environ, "LOG_DIR": f"{MOUNT_PATH}/checkpoints"},
            check=True,
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="A100-80GB:2",
    cpu=(8, 16),
    memory=(65536, 131072),
    timeout=60 * 60 * 14,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_model_ddp_remote(nproc: int = 2):
    """Runs Train.py across `nproc` GPUs on one machine using torchrun DDP."""
    import subprocess
    try:
        subprocess.run(
            ["torchrun", "--standalone", f"--nproc_per_node={nproc}", "Train.py"],
            cwd="/root/nanochat",
            env={**os.environ, "LOG_DIR": f"{MOUNT_PATH}/checkpoints"},
            check=True,
        )
    finally:
        volume.commit()


@app.local_entrypoint()
def main(stage: str = "data", nproc: int = 2):
    """Local CLI entrypoint: dispatches to the data-download or training function based on --stage."""
    if stage == "data":
        download_data_remote.remote()
    elif stage == "train":
        train_model_remote.remote()
    elif stage == "train_ddp":
        train_model_ddp_remote.remote(nproc=nproc)
    else:
        print(f"Unknown stage: {stage}")