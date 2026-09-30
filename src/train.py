# From: https://github.com/ashleve/lightning-hydra-template

import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0,1,2,3")
import socket
import subprocess
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import hydra
from omegaconf import DictConfig

try:
    import pyrootutils
except ModuleNotFoundError:
    pyrootutils = None


def _setup_project_root() -> Path:
    if pyrootutils is not None:
        return Path(pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True))

    current = Path(__file__).resolve()
    for parent in current.parents:
        if (parent / ".project-root").exists():
            if str(parent) not in sys.path:
                sys.path.insert(0, str(parent))
            os.chdir(parent)
            return parent
    raise FileNotFoundError(
        f"Could not find '.project-root' while bootstrapping from {current}."
    )


PROJECT_ROOT = _setup_project_root()

from src import utils

log = utils.get_pylogger(__name__)


def _running_on_juwels() -> bool:
    """Best-effort detection for JUWELS/JSC environments."""
    candidates = [
        os.getenv("SLURM_CLUSTER_NAME", ""),
        os.getenv("HOSTNAME", ""),
        os.getenv("SLURMD_NODENAME", ""),
        os.getenv("SLURM_JOB_NODELIST", ""),
        socket.gethostname(),
        socket.getfqdn(),
    ]
    text = " ".join(x.lower() for x in candidates if x)
    return "juwels" in text or "jsc" in text


def _fix_jsc_master_addr(num_nodes: int | None = None):
    """Use InfiniBand hostname for MASTER_ADDR on JUWELS multi-node jobs only."""
    if not _running_on_juwels():
        return

    if num_nodes is None:
        num_nodes = int(os.getenv("SLURM_NNODES", "1"))
    if int(num_nodes) <= 1:
        return

    nodelist = os.getenv("SLURM_JOB_NODELIST")
    if not nodelist:
        return

    master = subprocess.check_output(
        ["scontrol", "show", "hostnames", nodelist],
        text=True,
    ).splitlines()[0].strip()

    # Convert JUWELS hostname to InfiniBand hostname
    if master.endswith(".juwels"):
        master = master.replace(".juwels", "i.juwels")
    elif not master.endswith("i"):
        master = master + "i"

    os.environ["MASTER_ADDR"] = master
    os.environ.setdefault("MASTER_PORT", "12910")

    print(
        f"[startup][juwels] SLURM_JOB_NODELIST={nodelist} "
        f"MASTER_ADDR={os.environ['MASTER_ADDR']} "
        f"MASTER_PORT={os.environ['MASTER_PORT']}",
        flush=True,
    )


@utils.task_wrapper
def train(cfg: DictConfig) -> Tuple[dict, dict]:
    import torch
    import lightning as L
    from lightning import Callback, LightningDataModule, LightningModule, Trainer
    from lightning.pytorch.loggers import Logger

    utils.enable_trusted_checkpoint_resume()

    if cfg.get("seed"):
        L.seed_everything(cfg.seed, workers=True)

    torch.backends.cuda.enable_flash_sdp(True)
    torch.set_float32_matmul_precision("medium")

    log.info(f"Instantiating datamodule <{cfg.data._target_}>")
    datamodule: LightningDataModule = hydra.utils.instantiate(cfg.data)

    log.info(f"Instantiating model <{cfg.model._target_}>")
    model: LightningModule = hydra.utils.instantiate(cfg.model)
    if cfg.get("warmstart_ckpt_path"):
        log.info(
            "Warm-starting model weights only from checkpoint "
            f"<{cfg.warmstart_ckpt_path}>"
        )
        utils.load_model_weights_from_checkpoint(
            model=model,
            checkpoint_path=str(cfg.warmstart_ckpt_path),
            strict=bool(cfg.get("warmstart_strict", True)),
        )

    log.info("Instantiating callbacks...")
    callbacks: List[Callback] = utils.instantiate_callbacks(cfg.get("callbacks"))

    _extras = cfg.get("extras") or {}
    paths_cfg = cfg.get("paths") or {}
    paths_disable_wandb_osh = bool(paths_cfg.get("disable_wandb_osh", False))
    use_wandb_osh = _extras.get("use_wandb_osh", True)
    if paths_disable_wandb_osh:
        use_wandb_osh = False
        log.info("Disabling wandb-osh because the active paths profile requests it.")
    if (
        use_wandb_osh
        and cfg.get("logger")
        and cfg.logger.get("wandb")
        and cfg.logger.wandb.get("offline", False)
    ):
        try:
            from wandb_osh.lightning_hooks import TriggerWandbSyncLightningCallback

            comm_dir = _extras.get("wandb_osh_communication_dir") or str(cfg.paths.log_dir)
            callbacks.append(TriggerWandbSyncLightningCallback(communication_dir=comm_dir))
            log.info("Added TriggerWandbSyncLightningCallback for wandb-osh (offline sync)")
        except ImportError:
            log.warning(
                "wandb-osh not installed. Install with: pip install wandb-osh[lightning] "
                "for live sync when using wandb offline on compute nodes without internet."
            )

    log.info("Instantiating loggers...")
    logger: List[Logger] = utils.instantiate_loggers(cfg.get("logger"))

    trainer_cfg = cfg.get("trainer")
    trainer_num_nodes = 1
    if trainer_cfg is not None:
        trainer_num_nodes = int(trainer_cfg.get("num_nodes", 1))

    _fix_jsc_master_addr(num_nodes=trainer_num_nodes)

    print("ON NODE: torch.cuda.device_count() =", torch.cuda.device_count(), flush=True)
    print(
        f"HOST={socket.gethostname()} "
        f"RANK={os.getenv('RANK')} "
        f"LOCAL_RANK={os.getenv('LOCAL_RANK')} "
        f"SLURM_PROCID={os.getenv('SLURM_PROCID')} "
        f"CUDA_VISIBLE_DEVICES={os.getenv('CUDA_VISIBLE_DEVICES')} "
        f"MASTER_ADDR={os.getenv('MASTER_ADDR')} "
        f"MASTER_PORT={os.getenv('MASTER_PORT')}",
        flush=True,
    )

    log.info(f"Instantiating trainer <{cfg.trainer._target_}>")
    trainer_plugins = None
    trainer: Trainer = hydra.utils.instantiate(
        cfg.trainer,
        callbacks=callbacks,
        logger=logger,
        plugins=trainer_plugins,
    )

    object_dict = {
        "cfg": cfg,
        "datamodule": datamodule,
        "model": model,
        "callbacks": callbacks,
        "logger": logger,
        "trainer": trainer,
    }

    if logger:
        log.info("Logging hyperparameters!")
        utils.log_hyperparameters(object_dict)

    if cfg.get("train"):
        log.info("Starting training!")
        trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))

    train_metrics = trainer.callback_metrics

    if cfg.get("test"):
        log.info("Starting testing!")
        ckpt_path = trainer.checkpoint_callback.best_model_path
        if ckpt_path == "":
            log.warning("Best ckpt not found! Using current weights for testing...")
            ckpt_path = None
        trainer.test(model=model, datamodule=datamodule, ckpt_path=ckpt_path)
        log.info(f"Best ckpt path: {ckpt_path}")

    test_metrics = trainer.callback_metrics
    metric_dict = {**train_metrics, **test_metrics}

    return metric_dict, object_dict


@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
def main(cfg: DictConfig) -> Optional[float]:
    utils.extras(cfg)

    metric_dict, _ = train(cfg)

    metric_value = utils.get_metric_value(
        metric_dict=metric_dict, metric_name=cfg.get("optimized_metric")
    )
    return metric_value


if __name__ == "__main__":
    main()
