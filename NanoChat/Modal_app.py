"""
modal_app.py

Pulls the already-tokenized shards from HF into a Modal Volume, then
runs Train.py on a GPU against them. SFT reads its checkpoint + data
straight from the same volume.
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
    .add_local_file("train_utils.py", "/root/nanochat/train_utils.py")
    .add_local_file("Train.py", "/root/nanochat/Train.py")
    .add_local_file("train_sft.py", "/root/nanochat/train_sft.py")
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
            env={**os.environ, "LOG_DIR": f"{MOUNT_PATH}/checkpoints", "PYTHONUNBUFFERED": "1"},
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
            env={**os.environ, "LOG_DIR": f"{MOUNT_PATH}/checkpoints", "PYTHONUNBUFFERED": "1"},
            check=True,
        )
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="A100-80GB",
    cpu=(4, 8),
    memory=(32768, 65536),
    timeout=60 * 60 * 4,
    secrets=[modal.Secret.from_name("wandb-secret")],
)
def train_sft_remote(pretrain_checkpoint: str = "", data_path: str = ""):
    """Runs train_sft.py against a pretrained checkpoint + SFT data already sitting
    in the volume. Both args are optional -- leave blank to fall back to whatever
    Config.py's defaults resolve to (/data/log/model_checkpoint_step_02199.pt and
    /data/sft_conversations.jsonl respectively); pass them to override without
    touching Config.py. Timeout is a safety ceiling only -- Modal doesn't bill for
    unused timeout, only actual runtime, so it costs nothing to leave generous."""
    import subprocess
    env = {**os.environ, "SFT_LOG_DIR": f"{MOUNT_PATH}/checkpoints_sft", "PYTHONUNBUFFERED": "1"}
    if pretrain_checkpoint:
        env["SFT_PRETRAIN_CKPT"] = f"{MOUNT_PATH}/checkpoints/{pretrain_checkpoint}"
    if data_path:
        env["SFT_DATA_PATH"] = f"{MOUNT_PATH}/{data_path}"
    try:
        subprocess.run(["python", "train_sft.py"], cwd="/root/nanochat", env=env, check=True)
    finally:
        volume.commit()


@app.function(
    image=image,
    volumes=VOLUMES,
    gpu="A100-80GB",
    cpu=(4, 8),
    memory=(16384, 32768),
    timeout=60 * 60,
)
def final_eval_remote(checkpoint: str):
    """Runs the full (untrimmed) HellaSwag eval against a finished checkpoint."""
    import subprocess
    subprocess.run(
        ["python", "HellaSwag.py", "-c", f"{MOUNT_PATH}/checkpoints/{checkpoint}", "-t", f"{MOUNT_PATH}/tokenizer"],
        cwd="/root/nanochat",
        check=True,
    )


@app.local_entrypoint()
def main(stage: str = "data", nproc: int = 2, checkpoint: str = "", data_path: str = ""):
    """Local CLI entrypoint: dispatches to the data-download, training, SFT, or
    final-eval function based on --stage.

    Examples:
      modal run modal_app.py --stage data
      modal run modal_app.py --stage train
      modal run modal_app.py --stage train_ddp --nproc 2
      modal run modal_app.py --stage train_sft --checkpoint model_checkpoint_step_02199.pt
      modal run modal_app.py --stage train_sft --checkpoint model_checkpoint_step_02199.pt --data-path my_other_sft_set.jsonl
      modal run modal_app.py --stage final_eval --checkpoint model_checkpoint_step_02199.pt
    """
    if stage == "data":
        download_data_remote.remote()
    elif stage == "train":
        train_model_remote.remote()
    elif stage == "train_ddp":
        train_model_ddp_remote.remote(nproc=nproc)
    elif stage == "train_sft":
        train_sft_remote.remote(pretrain_checkpoint=checkpoint, data_path=data_path)
    elif stage == "final_eval":
        final_eval_remote.remote(checkpoint=checkpoint)
    else:
        print(f"Unknown stage: {stage}")