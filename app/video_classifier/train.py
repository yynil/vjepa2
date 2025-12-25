# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import gc
import random
import time

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from app.video_classifier.transforms import make_transforms
from app.video_classifier.utils import init_opt, init_video_model, load_checkpoint, load_pretrained_encoder
from src.datasets.data_manager import init_data
from src.models.attentive_pooler import AttentiveClassifier
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer

log_timings = True
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


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    cfgs_meta = args.get("meta")
    load_model = cfgs_meta.get("load_checkpoint") or resume_preempt
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

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    uniform_power = cfgs_model.get("uniform_power", False)
    use_rope = cfgs_model.get("use_rope", False)
    use_silu = cfgs_model.get("use_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    attn_pooler_depth = cfgs_model.get("attn_pooler_depth", 1)
    attn_pooler_heads = cfgs_model.get("attn_pooler_heads", None)

    # -- DATA
    cfgs_data = args.get("data")
    dataset_paths = cfgs_data.get("datasets", [])
    datasets_weights = cfgs_data.get("datasets_weights")
    dataset_fpcs = cfgs_data.get("dataset_fpcs")
    max_num_frames = max(dataset_fpcs)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), "Must have one sampling weight specified for each dataset"
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size")
    fps = cfgs_data.get("fps")
    crop_size = cfgs_data.get("crop_size", 224)
    patch_size = cfgs_data.get("patch_size")
    pin_mem = cfgs_data.get("pin_mem", False)
    num_workers = cfgs_data.get("num_workers", 1)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    frame_step = cfgs_data.get("frame_step", 4)
    num_clips = cfgs_data.get("num_clips", 1)

    # -- DATA AUGS
    cfgs_data_aug = args.get("data_aug")
    ar_range = cfgs_data_aug.get("random_resize_aspect_ratio", [3 / 4, 4 / 3])
    rr_scale = cfgs_data_aug.get("random_resize_scale", [0.3, 1.0])
    motion_shift = cfgs_data_aug.get("motion_shift", False)
    reprob = cfgs_data_aug.get("reprob", 0.0)
    use_aa = cfgs_data_aug.get("auto_augment", False)

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)
    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    latest_path = os.path.join(folder, "latest.pt")
    resume_path = os.path.join(folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    # -- make csv_logger
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

    # -- init model
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

    # data collator & transforms
    video_collator = _default_video_collator
    transform = make_transforms(
        random_horizontal_flip=cfgs_data_aug.get("horizontal_flip", False),
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # -- init data-loaders/samplers
    (train_loader, train_sampler) = init_data(
        data="videodataset",
        root_path=dataset_paths,
        frame_sample_rate=frame_step,
        clip_len=max_num_frames,
        fps=fps,
        dataset_fpcs=dataset_fpcs,
        num_clips=num_clips,
        random_clip_sampling=True,
        allow_clip_overlap=False,
        batch_size=batch_size,
        transform=transform,
        collator=video_collator,
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

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        classifier=classifier,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )
    encoder = DistributedDataParallel(encoder, static_graph=True)
    classifier = DistributedDataParallel(classifier, static_graph=True)

    # -- load pretrained weights
    if p_file is not None and load_encoder:
        encoder = load_pretrained_encoder(
            r_path=p_file,
            encoder=encoder,
        )

    start_epoch = 0
    # -- load training checkpoint
    if load_model and resume_path is not None and os.path.exists(resume_path):
        encoder, classifier, optimizer, scaler, start_epoch = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
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

    # -- TRAINING LOOP
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

            def load_batch():
                videos = sample[0].to(device, non_blocking=True)  # [B num_clips C T H W]
                labels = sample[1].to(device, dtype=torch.long)
                return videos, labels

            videos, labels = load_batch()
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                logger.info("Running garbage collection...")
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def _predict_logits(videos):
                    bsz, n_clips = videos.shape[:2]
                    flat_videos = videos.view(-1, *videos.shape[2:])
                    tokens = encoder(flat_videos)
                    logits = classifier(tokens)
                    logits = logits.view(bsz, n_clips, -1).mean(dim=1)
                    return logits

                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    logits = _predict_logits(videos)
                    loss = F.cross_entropy(logits, labels)

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

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

        # -- Save Checkpoint
        logger.info("avg. loss %.3f" % loss_meter.avg)
        # -- Save Last
        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f"e{epoch}.pt"
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
