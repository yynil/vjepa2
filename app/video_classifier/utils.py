# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys
from typing import Iterable, Optional, Sequence, Tuple

import torch

import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def load_pretrained_encoder(
    r_path: str,
    encoder: torch.nn.Module,
    context_encoder_key: str = "encoder",
    load_encoder: bool = True,
) -> torch.nn.Module:
    logger.info(f"Loading pretrained encoder from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint.get("epoch", "NA")

    if load_encoder:
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    del checkpoint
    return encoder


def load_checkpoint(
    r_path: str,
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    opt: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    replace_kw: Sequence[str] = ("backbone.",),
) -> Tuple[torch.nn.Module, torch.nn.Module, Optional[torch.optim.Optimizer], Optional[torch.cuda.amp.GradScaler], int]:
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))

    epoch = checkpoint["epoch"]

    def _clean_state_dict(pretrained_dict):
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        return pretrained_dict

    pretrained_dict = _clean_state_dict(checkpoint["encoder"])
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if "classifier" in checkpoint:
        msg = classifier.load_state_dict(checkpoint["classifier"], strict=False)
        logger.info(f"loaded pretrained classifier from epoch {epoch} with msg: {msg}")

    if opt is not None and "opt" in checkpoint:
        opt.load_state_dict(checkpoint["opt"])

    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])

    logger.info(f"read-path: {r_path}")
    del checkpoint

    return encoder, classifier, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=16,
    tubelet_size=2,
    model_name="vit_base",
    crop_size=224,
    uniform_power=False,
    use_sdpa=False,
    use_rope=False,
    use_silu=False,
    wide_silu=False,
    use_activation_checkpointing=False,
):
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
    )

    encoder.to(device)
    logger.info(encoder)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters (trainable): {count_parameters(encoder)}")

    return encoder


def _named_params_excluding_bias_and_norm(module: torch.nn.Module, lr_scale: Optional[float] = None):
    group = {
        "params": (p for n, p in module.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
    }
    if lr_scale is not None:
        group["lr_scale"] = lr_scale
    return group


def _named_params_bias_and_norm(module: torch.nn.Module, lr_scale: Optional[float] = None, zero_init_bias_wd=True):
    group = {
        "params": (p for n, p in module.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
        "WD_exclude": zero_init_bias_wd,
        "weight_decay": 0,
    }
    if lr_scale is not None:
        group["lr_scale"] = lr_scale
    return group


def init_opt(
    encoder: torch.nn.Module,
    classifier: torch.nn.Module,
    iterations_per_epoch: int,
    start_lr: float,
    ref_lr: float,
    warmup: float,
    anneal: float,
    num_epochs: int,
    wd: float = 1e-6,
    final_wd: float = 1e-6,
    final_lr: float = 0.0,
    mixed_precision: bool = False,
    betas: Iterable[float] = (0.9, 0.999),
    eps: float = 1e-8,
    zero_init_bias_wd: bool = True,
    enc_lr_scale: float = 1.0,
):
    param_groups = []

    if any(p.requires_grad for p in encoder.parameters()):
        param_groups.extend(
            [
                _named_params_excluding_bias_and_norm(encoder, lr_scale=enc_lr_scale),
                _named_params_bias_and_norm(encoder, lr_scale=enc_lr_scale, zero_init_bias_wd=zero_init_bias_wd),
            ]
        )

    param_groups.extend(
        [
            _named_params_excluding_bias_and_norm(classifier),
            _named_params_bias_and_norm(classifier, zero_init_bias_wd=zero_init_bias_wd),
        ]
    )

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
