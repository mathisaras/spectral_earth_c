from __future__ import annotations

import inspect
from pathlib import Path
from typing import Any, Optional

import torch
from torch import nn

from src.utils.pylogger import get_pylogger

log = get_pylogger(__name__)

_PATCH_SENTINEL = "_spectral_earth_trusted_resume_patch"


def _torch_load_trusted(
    path_or_file: Any,
    map_location: Any = None,
    weights_only: Optional[bool] = None,
    **pickle_load_args: Any,
) -> Any:
    """Load a trusted checkpoint with full pickle support.

    PyTorch 2.6 changed ``torch.load`` to default to ``weights_only=True``.
    Lightning's checkpoint resume path relies on loading full trainer state,
    which includes OmegaConf objects in our checkpoints. We explicitly force
    ``weights_only=False`` when the runtime supports it.
    """
    kwargs = {"map_location": map_location, **pickle_load_args}
    try:
        if "weights_only" in inspect.signature(torch.load).parameters:
            kwargs["weights_only"] = False if weights_only is None else weights_only
    except (TypeError, ValueError):
        # Some wrapped/built-in implementations may not expose a signature.
        pass
    return torch.load(path_or_file, **kwargs)


def enable_trusted_checkpoint_resume() -> None:
    """Patch Lightning/Fabric checkpoint loading for trusted local resume.

    This keeps resume and best-checkpoint reloads working across Torch versions,
    especially on Torch >= 2.6 where ``torch.load`` defaults to
    ``weights_only=True``.
    """

    serialization = getattr(torch, "serialization", None)
    add_safe_globals = getattr(serialization, "add_safe_globals", None)
    if callable(add_safe_globals):
        try:
            from omegaconf import DictConfig, ListConfig
            from omegaconf.base import ContainerMetadata

            add_safe_globals([DictConfig, ListConfig, ContainerMetadata])
        except Exception as exc:
            log.warning(f"Could not register OmegaConf safe globals for checkpoint loading: {exc}")

    try:
        import lightning_fabric.plugins.io.torch_io as fabric_torch_io
        import lightning_fabric.utilities.cloud_io as fabric_cloud_io
        from lightning_fabric.utilities.cloud_io import get_filesystem
    except Exception as exc:
        log.warning(f"Could not patch Lightning checkpoint loading: {exc}")
        return

    if getattr(fabric_torch_io.pl_load, _PATCH_SENTINEL, False):
        return

    def _trusted_pl_load(
        path_or_url: Any,
        map_location: Optional[Any] = None,
        weights_only: Optional[bool] = None,
        **pickle_load_args: Any,
    ) -> Any:
        if not isinstance(path_or_url, (str, Path)):
            return _torch_load_trusted(
                path_or_url,
                map_location=map_location,
                weights_only=weights_only,
                **pickle_load_args,
            )

        if str(path_or_url).startswith("http"):
            return torch.hub.load_state_dict_from_url(
                str(path_or_url),
                map_location=map_location,
                weights_only=False if weights_only is None else weights_only,
            )

        fs = get_filesystem(path_or_url)
        with fs.open(path_or_url, "rb") as f:
            return _torch_load_trusted(
                f,
                map_location=map_location,
                weights_only=weights_only,
                **pickle_load_args,
            )

    original_load_checkpoint = fabric_torch_io.TorchCheckpointIO.load_checkpoint

    def _trusted_load_checkpoint(
        self: Any,
        path: Any,
        map_location: Optional[Any] = lambda storage, loc: storage,
        weights_only: Optional[bool] = None,
    ) -> dict[str, Any]:
        return original_load_checkpoint(
            self,
            path,
            map_location=map_location,
            weights_only=False if weights_only is None else weights_only,
        )

    setattr(_trusted_pl_load, _PATCH_SENTINEL, True)
    setattr(_trusted_load_checkpoint, _PATCH_SENTINEL, True)
    fabric_torch_io.pl_load = _trusted_pl_load
    fabric_cloud_io._load = _trusted_pl_load
    fabric_torch_io.TorchCheckpointIO.load_checkpoint = _trusted_load_checkpoint


def load_model_weights_from_checkpoint(
    model: nn.Module,
    checkpoint_path: str,
    strict: bool = True,
    map_location: Any = "cpu",
) -> None:
    """Load only model weights from a Lightning checkpoint.

    This intentionally avoids Lightning's ``ckpt_path`` restore path, so trainer
    loop state, optimizer state, and scheduler state are not restored.
    """

    ckpt_path = Path(str(checkpoint_path)).expanduser()
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"warmstart_ckpt_path does not exist: '{ckpt_path}'")

    checkpoint = _torch_load_trusted(str(ckpt_path), map_location=map_location)
    if isinstance(checkpoint, dict) and isinstance(checkpoint.get("state_dict"), dict):
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(
            "Unsupported checkpoint format for warmstart_ckpt_path. "
            "Expected a state_dict or Lightning checkpoint containing 'state_dict'."
        )

    state_dict = {str(k): v for k, v in state_dict.items() if torch.is_tensor(v)}
    incompatible = model.load_state_dict(state_dict, strict=bool(strict))
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    print(
        "[checkpoint warmstart] loaded model weights only from "
        f"'{ckpt_path}' strict={bool(strict)} "
        f"missing={len(missing)} unexpected={len(unexpected)}"
    )
    if missing:
        print(f"[checkpoint warmstart] missing keys sample: {missing[:20]}")
    if unexpected:
        print(f"[checkpoint warmstart] unexpected keys sample: {unexpected[:20]}")
