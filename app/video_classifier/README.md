# Video Classifier

This document summarizes the training pipeline, data expectations, model components, and launch settings for `app/video_classifier`.

## Data format and loading
- **Input listing**: Provide one or more CSV files via `data.datasets`, each line formatted as `path/to/video.mp4 <label_id>`. NPY files containing video paths are also accepted (labels default to `0`).【F:configs/train/vitb16/video-classifier.yaml†L6-L32】【F:src/datasets/video_dataset.py†L61-L111】
- **Frame sampling**: Each dataset specifies `frames_per_clip` through `dataset_fpcs`; per-clip sampling stride is defined by exactly one of `frame_step`, `fps`, or `duration`. `num_clips` controls how many clips are drawn per video segment.【F:src/datasets/video_dataset.py†L34-L79】【F:src/datasets/video_dataset.py†L186-L243】
- **Loader composition**: `init_data(data="videodataset")` builds a `DataLoader` with a distributed sampler (optionally weighted). Batches are collated into `[B, num_clips, C, T, H, W]` tensors plus labels and clip indices.【F:src/datasets/data_manager.py†L12-L69】【F:app/video_classifier/train.py†L25-L55】
- **Augmentations & tensorization** (`app/vjepa/transforms.py`):
  - Optional AutoAugment (RandAugment variant) or direct float conversion.
  - Random resized crop (with optional motion shift), random horizontal flip.
  - Channel-wise normalization; optional Random Erasing.
  - Output per clip: `[C, T, H, W]` float tensor.【F:app/vjepa/transforms.py†L13-L111】【F:app/vjepa/transforms.py†L113-L158】

## Model architecture
- **Encoder**: `VisionTransformer` (video mode uses `PatchEmbed3D`), with 3D sin-cos positional encoding or RoPE. Configurable patch size, tubelet size, depth/heads, activation (GELU/SiLU), SDPA, and activation checkpointing.【F:src/models/vision_transformer.py†L13-L118】【F:src/models/vision_transformer.py†L170-L236】
- **Classifier**: `AttentiveClassifier` applies `AttentivePooler` cross-attention pooling over encoder tokens, then a linear head to `num_classes`. Pooler depth/heads are configurable.【F:src/models/attentive_pooler.py†L10-L97】【F:src/models/attentive_pooler.py†L100-L129】
- **Clip fusion**: For `num_clips > 1`, logits are averaged across clips before loss computation.【F:app/video_classifier/train.py†L123-L157】

## Training loop
- **Loss & metrics**: Cross-entropy loss; top-1/top-5 accuracy tracked per iteration. Mixed precision is supported via `dtype` (bf16/fp16) and `GradScaler`.【F:app/video_classifier/train.py†L152-L211】【F:app/video_classifier/train.py†L238-L307】
- **Optimization**: AdamW with grouped params (norm/bias excluded from weight decay). Learning-rate schedule `WSDSchedule` (warmup + cosine-style decay) and weight-decay schedule `CosineWDSchedule`. Optional encoder LR scaling via `enc_lr_scale`.【F:app/video_classifier/utils.py†L73-L165】
- **Checkpointing & logging**: CSV logs per rank (`log_r{rank}.csv`). Checkpoints saved every epoch (`latest.pt`) and optionally at `save_every_freq`. Resuming supports loading encoder/classifier/optimizer/scaler state.【F:app/video_classifier/train.py†L58-L118】【F:app/video_classifier/train.py†L309-L375】

## Key configuration knobs
- **Data**: `batch_size`, `crop_size`, `patch_size`, `tubelet_size`, `fps/frame_step/duration`, `num_clips`, `dataset_fpcs`, `num_workers`, `pin_mem`.【F:configs/train/vitb16/video-classifier.yaml†L6-L32】
- **Model**: `model_name` (e.g., `vit_base`), `uniform_power`, `use_rope`, `use_silu`, `wide_silu`, `attn_pooler_depth`, `attn_pooler_heads`, `use_activation_checkpointing`, `compile_model`.【F:configs/train/vitb16/video-classifier.yaml†L34-L52】
- **Meta**: `num_classes`, `dtype`, `load_encoder` / `finetune_encoder`, `pretrain_checkpoint`, `load_checkpoint`, `seed`, `use_sdpa`.【F:configs/train/vitb16/video-classifier.yaml†L34-L52】
- **Optimization**: `epochs`, `ipe` (iterations per epoch; defaults to loader length), `lr`, `start_lr`, `final_lr`, `warmup`, `anneal`, `weight_decay`, `final_weight_decay`, `enc_lr_scale`, `betas`, `eps`.【F:configs/train/vitb16/video-classifier.yaml†L54-L69】

## Example launch
```bash
python -m torch.distributed.launch --nproc_per_node=<gpus> \
    app/video_classifier/train.py \
    --cfg configs/train/vitb16/video-classifier.yaml \
    --folder /your_folder/video_classifier/vitb16-224px-8f
```
- Replace CSV and `pretrain_checkpoint` paths with your data/checkpoints.
- Set `finetune_encoder: true` in the config to train the encoder; otherwise only the classifier is updated.【F:app/video_classifier/train.py†L94-L136】
