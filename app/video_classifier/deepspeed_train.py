import argparse
import gc
import os
import random
import time
from pathlib import Path

import deepspeed
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml

from app.video_classifier.transforms import make_transforms
from app.video_classifier.utils import (
    _named_params_bias_and_norm,
    _named_params_excluding_bias_and_norm,
    init_video_model,
    load_checkpoint,
    load_pretrained_encoder,
)
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

log_freq = 10
CHECKPOINT_FREQ = 1
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__, force=True)


def _default_video_collator(batch):
    videos, labels, clip_indices = zip(*batch)
    num_clips = len(videos[0])
    video_tensor = torch.stack([torch.stack(v, dim=0) for v in videos], dim=0)
    labels = torch.tensor(labels, dtype=torch.long)
    return video_tensor, labels, clip_indices


def _ensure_nccl_p2p_compatible():
    """Disable NCCL P2P if any visible GPU pair lacks peer access."""
    if not torch.cuda.is_available():
        return

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices is None:
        device_ids = list(range(torch.cuda.device_count()))
    else:
        try:
            device_ids = [int(d.strip()) for d in visible_devices.split(",")]
        except ValueError:
            device_ids = list(range(torch.cuda.device_count()))

    for i, src in enumerate(device_ids):
        for dst in device_ids[i + 1 :]:
            if (not torch.cuda.can_device_access_peer(src, dst)) or (not torch.cuda.can_device_access_peer(dst, src)):
                if os.environ.get("NCCL_P2P_DISABLE") != "1":
                    logger.warning(
                        "Disabling NCCL P2P because peer access is not available between visible devices "
                        f"{src} and {dst}."
                    )
                    os.environ["NCCL_P2P_DISABLE"] = "1"
                return


class VideoClassificationModel(nn.Module):
    def __init__(self, encoder: nn.Module, classifier: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def forward(self, videos: torch.Tensor) -> torch.Tensor:
        bsz, n_clips = videos.shape[:2]
        flat_videos = videos.view(-1, *videos.shape[2:])
        tokens = self.encoder(flat_videos)
        logits = self.classifier(tokens)
        logits = logits.view(bsz, n_clips, -1).mean(dim=1)
        return logits


def _resolve_frame_sampling(name, fps, duration, frame_step, default_frame_step=4):
    """Ensure exactly one of fps, duration, or frame_step is set for a loader."""

    specified = [v is not None for v in (fps, duration, frame_step)]
    if sum(specified) > 1:
        logger.warning(
            f"Multiple sampling options set for {name}; prioritizing in order fps > duration > frame_step: "
            f"fps={fps}, duration={duration}, frame_step={frame_step}."
        )
    if fps is not None:
        duration = None
        frame_step = None
    elif duration is not None:
        frame_step = None
    elif frame_step is None:
        frame_step = default_frame_step
    return fps, duration, frame_step


def _build_deepspeed_config(
    batch_size_per_gpu: int,
    world_size: int,
    dtype: torch.dtype,
    stage: int,
    grad_accum_steps: int,
) -> dict:
    config = {
        "train_batch_size": batch_size_per_gpu * world_size * grad_accum_steps,
        "train_micro_batch_size_per_gpu": batch_size_per_gpu,
        "gradient_accumulation_steps": grad_accum_steps,
        "zero_optimization": {
            "stage": stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
        "gradient_clipping": 0.0,
        "steps_per_print": 0,
        "wall_clock_breakdown": False,
    }

    if dtype == torch.bfloat16:
        config["bf16"] = {"enabled": True}
    elif dtype == torch.float16:
        config["fp16"] = {"enabled": True, "loss_scale": 0}

    return config


def _init_optimizer(
    model: nn.Module,
    optimizer_cfg: dict,
    use_mixed_precision: bool,
):
    params = []
    encoder, classifier = model.encoder, model.classifier
    if any(p.requires_grad for p in encoder.parameters()):
        params.extend(
            [
                _named_params_excluding_bias_and_norm(encoder, lr_scale=optimizer_cfg.get("enc_lr_scale", 1.0)),
                _named_params_bias_and_norm(encoder, lr_scale=optimizer_cfg.get("enc_lr_scale", 1.0)),
            ]
        )

    params.extend(
        [
            _named_params_excluding_bias_and_norm(classifier),
            _named_params_bias_and_norm(classifier),
        ]
    )

    optimizer = torch.optim.AdamW(params, betas=optimizer_cfg.get("betas", (0.9, 0.999)), eps=optimizer_cfg["eps"])

    return optimizer


def _build_schedulers(optimizer: torch.optim.Optimizer, optimizer_cfg: dict, iterations_per_epoch: int):
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(optimizer_cfg["warmup"] * iterations_per_epoch),
        anneal_steps=int(optimizer_cfg["anneal"] * iterations_per_epoch),
        start_lr=optimizer_cfg["start_lr"],
        ref_lr=optimizer_cfg["lr"],
        final_lr=optimizer_cfg["final_lr"],
        T_max=int(optimizer_cfg["epochs"] * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=float(optimizer_cfg["weight_decay"]),
        final_wd=float(optimizer_cfg["final_weight_decay"]),
        T_max=int(optimizer_cfg["epochs"] * iterations_per_epoch),
    )
    return scheduler, wd_scheduler


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main(config: dict, deepspeed_stage: int = 2, grad_accum_steps: int = 1):
    folder = config.get("folder")
    cfgs_meta = config.get("meta", {})
    cfgs_model = config.get("model", {})
    cfgs_data = config.get("data", {})
    cfgs_data_aug = config.get("data_aug", {})
    cfgs_opt = config.get("optimization", {})

    os.makedirs(folder, exist_ok=True)

    load_model = cfgs_meta.get("load_checkpoint", False)
    r_file = cfgs_meta.get("read_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_encoder = cfgs_meta.get("load_encoder", True)
    finetune_encoder = cfgs_meta.get("finetune_encoder", False)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", False)
    sync_gc = cfgs_meta.get("sync_gc", False)
    num_classes = cfgs_meta.get("num_classes")
    if num_classes is None:
        raise ValueError("num_classes must be specified in meta configuration.")
    which_dtype = cfgs_meta.get("dtype", "float32")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    attn_pooler_depth = cfgs_model.get("attn_pooler_depth", 1)
    attn_pooler_heads = cfgs_model.get("attn_pooler_heads", None)

    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    duration = cfgs_data.get("duration")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    frame_step = cfgs_data.get("frame_step")
    fps, duration, frame_step = _resolve_frame_sampling("training", fps, duration, frame_step, default_frame_step=4)
    num_clips = cfgs_data.get("num_clips", 1)
    val_dataset_paths = cfgs_data.get("val_datasets")
    val_dataset_fpcs = cfgs_data.get("val_dataset_fpcs", dataset_fpcs)
    val_datasets_weights = cfgs_data.get("val_datasets_weights")
    eval_max_num_frames = max(val_dataset_fpcs) if val_dataset_fpcs is not None else max_num_frames
    val_batch_size = cfgs_data.get("val_batch_size", batch_size)
    val_num_workers = cfgs_data.get("val_num_workers", num_workers)
    val_persistent_workers = cfgs_data.get("val_persistent_workers", persistent_workers)
    val_fps = cfgs_data.get("val_fps", fps)
    val_duration = cfgs_data.get("val_duration")
    val_frame_step = cfgs_data.get("val_frame_step")
    val_fps, val_duration, val_frame_step = _resolve_frame_sampling(
        "validation",
        val_fps,
        val_duration,
        val_frame_step,
        default_frame_step=frame_step if frame_step is not None else 4,
    )
    val_num_clips = cfgs_data.get("val_num_clips", num_clips)
    val_random_clip_sampling = cfgs_data.get("val_random_clip_sampling", False)

    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    ipe = cfgs_opt.get("ipe", None)
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        torch.multiprocessing.set_start_method("spawn")
    except Exception:
        pass

    _ensure_nccl_p2p_compatible()

    if not torch.distributed.is_initialized():
        deepspeed.init_distributed()
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "acc@1"),
        ("%.5f", "acc@5"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    encoder = init_video_model(
        uniform_power=uniform_power,
        device=device,
        patch_size=patch_size,
        max_num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        use_sdpa=use_sdpa,
        use_rope=use_rope,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
    )
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=attn_pooler_heads if attn_pooler_heads is not None else encoder.num_heads,
        depth=attn_pooler_depth,
        num_classes=num_classes,
        use_activation_checkpointing=use_activation_checkpointing,
    ).to(device)
    if not finetune_encoder:
        for p in encoder.parameters():
            p.requires_grad = False

    if compile_model:
        logger.info("Compiling encoder and classifier.")
        torch._dynamo.config.optimize_ddp = False
        encoder.compile()
        classifier.compile()

    model = VideoClassificationModel(encoder=encoder, classifier=classifier)

    transform = make_transforms(
        random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )
    eval_transform = None
    if val_dataset_paths:
        eval_transform = make_transforms(
            random_horizontal_flip=False,
            random_resize_aspect_ratio=ar_range,
            random_resize_scale=rr_scale,
            reprob=0.0,
            auto_augment=False,
            motion_shift=False,
            crop_size=crop_size,
        )

    val_loader = None
    val_sampler = None
    (train_loader, train_sampler) = init_data(
        data="videodataset",
        root_path=dataset_paths,
        frame_sample_rate=frame_step,
        clip_len=max_num_frames,
        fps=fps,
        duration=duration,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        batch_size=batch_size,
        transform=transform,
        collator=_default_video_collator,
        num_workers=num_workers,
        world_size=world_size,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        rank=rank,
        datasets_weights=datasets_weights,
    )
    _dlen = len(train_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")
    if val_dataset_paths:
        (val_loader, val_sampler) = init_data(
            data="videodataset",
            root_path=val_dataset_paths,
            frame_sample_rate=val_frame_step,
            clip_len=eval_max_num_frames,
            fps=val_fps,
            duration=val_duration,
            dataset_fpcs=val_dataset_fpcs,
            num_clips=val_num_clips,
            random_clip_sampling=val_random_clip_sampling,
            allow_clip_overlap=False,
            batch_size=val_batch_size,
            transform=eval_transform if eval_transform is not None else transform,
            collator=_default_video_collator,
            num_workers=val_num_workers,
            world_size=world_size,
            pin_mem=pin_mem,
            persistent_workers=val_persistent_workers,
            rank=rank,
            datasets_weights=val_datasets_weights,
            training=False,
            drop_last=False,
        )
        logger.info(f"Validation loader initialized with length {len(val_loader)}")

    ds_config = _build_deepspeed_config(
        batch_size_per_gpu=batch_size,
        world_size=world_size,
        dtype=dtype,
        stage=deepspeed_stage,
        grad_accum_steps=grad_accum_steps,
    )

    optimizer_cfg = {
        "start_lr": start_lr,
        "lr": lr,
        "final_lr": final_lr,
        "warmup": warmup,
        "anneal": anneal,
        "epochs": num_epochs,
        "weight_decay": wd,
        "final_weight_decay": final_wd,
        "eps": eps,
        "betas": betas,
        "enc_lr_scale": enc_lr_scale,
    }

    optimizer = _init_optimizer(
        model=model,
        optimizer_cfg=optimizer_cfg,
        use_mixed_precision=mixed_precision,
    )

    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=[p for p in model.parameters() if p.requires_grad],
        optimizer=optimizer,
        lr_scheduler=None,
        config=ds_config,
    )

    # rebuild schedulers with the DeepSpeed optimizer to keep param group references in sync
    scheduler, wd_scheduler = _build_schedulers(optimizer=optimizer, optimizer_cfg=optimizer_cfg, iterations_per_epoch=ipe)

    if p_file is not None and load_encoder:
        model_engine.module.encoder = load_pretrained_encoder(
            r_path=p_file,
            encoder=model_engine.module.encoder,
        )

    start_epoch = 0
    if load_model and resume_path is not None and os.path.exists(resume_path):
        (
            model_engine.module.encoder,
            model_engine.module.classifier,
            optimizer,
            _,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=model_engine.module.encoder,
            classifier=model_engine.module.classifier,
            opt=optimizer,
            scaler=None,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": model_engine.module.encoder.state_dict(),
            "classifier": model_engine.module.classifier.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None,
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")

    logger.info("Initializing loader...")
    train_sampler.set_epoch(start_epoch)
    loader = iter(train_loader)

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(train_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    def evaluate(epoch):
        if val_loader is None:
            return None

        model_engine.eval()

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()

        if val_sampler is not None:
            val_sampler.set_epoch(epoch)

        with torch.inference_mode():
            for sample in val_loader:
                videos = sample[0].to(device, non_blocking=True)
                labels = sample[1].to(device, dtype=torch.long)
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    logits = model_engine(videos)
                    loss = F.cross_entropy(logits, labels)

                maxk = min(5, num_classes)
                _, pred = logits.topk(maxk, 1, True, True)
                correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                acc1 = correct[:, :1].reshape(-1).float().mean().item()
                acc5 = correct[:, :maxk].reshape(-1).float().mean().item()

                loss_meter.update(float(loss))
                acc1_meter.update(acc1)
                acc5_meter.update(acc5)

        if rank == 0:
            logger.info(
                "[Eval][%d] loss: %.3f [acc@1: %.3f acc@5: %.3f]"
                % (epoch + 1, loss_meter.avg, acc1_meter.avg, acc5_meter.avg)
            )

        model_engine.train()

        return loss_meter.avg, acc1_meter.avg, acc5_meter.avg

    if num_epochs == 0 and val_loader is not None:
        evaluate(start_epoch)
        return

    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))

        loss_meter = AverageMeter()
        acc1_meter = AverageMeter()
        acc5_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    train_sampler.set_epoch(epoch)
                    loader = iter(train_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Encountered exception when loading data (num retries {iter_retries}):\n{e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        logger.warning(f"Exceeded max retries ({NUM_RETRIES}) when loading data. Skipping batch.")
                        raise e

            videos = sample[0].to(device, non_blocking=True)
            labels = sample[1].to(device, dtype=torch.long)
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    logits = model_engine(videos)
                    loss = F.cross_entropy(logits, labels)

                model_engine.backward(loss)
                model_engine.step()

                with torch.no_grad():
                    maxk = min(5, num_classes)
                    _, pred = logits.topk(maxk, 1, True, True)
                    correct = pred.eq(labels.view(-1, 1).expand_as(pred))
                    acc1 = correct[:, :1].reshape(-1).float().mean().item()
                    acc5 = correct[:, :maxk].reshape(-1).float().mean().item()

                return (
                    float(loss),
                    acc1,
                    acc5,
                    _new_lr,
                    _new_wd,
                )

            (
                loss,
                acc1,
                acc5,
                _new_lr,
                _new_wd,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0
            loss_meter.update(loss)
            acc1_meter.update(acc1)
            acc5_meter.update(acc5)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    acc1,
                    acc5,
                    iter_elapsed_time_ms,
                    gpu_etime_ms,
                    data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [acc@1: %.3f acc@5: %.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] "
                        "[gpu: %.1f ms] "
                        "[data: %.1f ms]"
                        % (
                            epoch + 1,
                            itr,
                            loss_meter.avg,
                            acc1_meter.avg,
                            acc5_meter.avg,
                            _new_wd,
                            _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2 if torch.cuda.is_available() else 0.0,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        logger.info("avg. loss %.3f" % loss_meter.avg)
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)

        if val_loader is not None:
            evaluate(epoch)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSpeed entrypoint for video classifier training.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("video-classifier.yaml")),
        help="Path to the video-classifier YAML config.",
    )
    parser.add_argument("--deepspeed-stage", type=int, default=2, help="ZeRO optimization stage (default: 2).")
    parser.add_argument(
        "--grad-accum-steps", type=int, default=1, help="Gradient accumulation steps for DeepSpeed."
    )
    return parser.parse_args()


if __name__ == "__main__":
    # -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
    except Exception:
        pass

    args = parse_args()
    cfg = _load_config(args.config)
    main(cfg, deepspeed_stage=args.deepspeed_stage, grad_accum_steps=args.grad_accum_steps)
