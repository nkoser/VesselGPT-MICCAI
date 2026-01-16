import argparse
import glob
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    import wandb
except Exception:
    wandb = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from Stage1.modelsMultitalk.stage1_vocaset import VQAutoEncoder
from Stage1.base.utilities import AverageMeter
from funciones import IntraDataset, save_best_model, Args
from Stage1.metrics.loss import calc_vq_loss


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}

def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device_value):
    if isinstance(device_value, int):
        device_value = f"cuda:{device_value}"
    if isinstance(device_value, str) and device_value.startswith("cuda"):
        if torch.cuda.is_available():
            print(f"Using device: {torch.cuda.get_device_name(0)}")
            return torch.device(device_value)
    print("CUDA not available. Using CPU.")
    return torch.device("cpu")


def iter_files(folder):
    return [os.path.basename(x) for x in glob.glob(os.path.join(folder, "*.npy"))]


def build_loader(
    folder,
    mode,
    batch_size,
    shuffle,
    num_workers=0,
    pin_memory=False,
    persistent_workers=False,
    prefetch_factor=None,
):
    files = iter_files(folder)
    dataset = IntraDataset(files, root_dir=folder, mode=mode)
    loader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "persistent_workers": persistent_workers if num_workers > 0 else False,
    }
    if prefetch_factor is not None and num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
    loader = DataLoader(dataset, **loader_kwargs)
    return loader


def calc_vq_loss_masked(pred, target, quant_loss, quant_loss_weight, mask=None):
    if mask is None:
        return calc_vq_loss(pred, target, quant_loss=quant_loss, quant_loss_weight=quant_loss_weight)
    diff = torch.abs(pred - target)
    valid = ~mask.unsqueeze(-1)
    if valid.any():
        rec_loss = diff[valid].mean()
    else:
        rec_loss = torch.zeros((), device=pred.device)
    total_loss = rec_loss + quant_loss_weight * quant_loss
    return total_loss, (rec_loss, quant_loss)


def train_one_epoch(model, data_loader, optimizer, device, quant_loss_weight, mask_null=False, null_threshold=1e-3):
    model.train()
    rec_loss_meter = AverageMeter()
    quant_loss_meter = AverageMeter()
    total_loss_meter = AverageMeter()
    pp_meter = AverageMeter()

    for inputs in data_loader:
        inputs = inputs.to(device)
        if inputs.shape[1] <= 1:
            continue
        out, quant_loss, info = model(inputs)
        mask = None
        if mask_null:
            mask = torch.all(torch.abs(inputs) <= null_threshold, dim=2)
        loss, loss_details = calc_vq_loss_masked(
            out, inputs, quant_loss=quant_loss, quant_loss_weight=quant_loss_weight, mask=mask
        )
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        rec_loss_meter.update(loss_details[0].item(), 1)
        quant_loss_meter.update(loss_details[1].item(), 1)
        total_loss_meter.update(loss.item(), 1)
        if info is not None and len(info) > 0:
            pp_meter.update(info[0].mean().item(), 1)

    return total_loss_meter.avg, rec_loss_meter.avg, quant_loss_meter.avg, pp_meter.avg


def validate(val_loader, model, device, quant_loss_weight, epoch=0, mask_null=False, null_threshold=1e-3):
    accumulated_loss = 0.0
    accumulated_rec = 0.0
    accumulated_quant = 0.0
    model.eval()

    with torch.no_grad():
        for idx, inputs in enumerate(val_loader):
            inputs = inputs.to(device)
            if inputs.shape[1] <= 1:
                continue
            out, quant_loss, _info = model(inputs)
            mask = None
            if mask_null:
                mask = torch.all(torch.abs(inputs) <= null_threshold, dim=2)
            loss, loss_details = calc_vq_loss_masked(
                out, inputs, quant_loss=quant_loss, quant_loss_weight=quant_loss_weight, mask=mask
            )
            accumulated_loss += loss
            accumulated_rec += loss_details[0]
            accumulated_quant += loss_details[1]

            # save the reconstructed trees for visualization (mask zeros from input)
            if epoch % 20 == 0:
                save_dir = Path("Stage1_New/reconstructions_stage1")
                save_dir.mkdir(parents=True, exist_ok=True)
                for i in range(inputs.shape[0]):
                    input_tree = inputs[i].detach().cpu().numpy()
                    recon_tree = out[i].detach().cpu().numpy()
                    mask = np.all(np.abs(input_tree) <= 1e-3, axis=1)
                    recon_tree[mask] = 0
                    recon_tree[np.abs(recon_tree) < 1e-3] = 0
                    np.save(save_dir / f"input_epoch_{epoch + 1}_batch_{i}_sample_{idx}.npy", input_tree)
                    np.save(save_dir / f"recon_epoch_{epoch + 1}_batch_{i}_sample_{idx}.npy", recon_tree)

    avg_loss = accumulated_loss / len(val_loader)
    rec_loss = accumulated_rec / len(val_loader)
    quant_loss = accumulated_quant / len(val_loader)
    return avg_loss, rec_loss, quant_loss


def main():
    parser = argparse.ArgumentParser(description="Train Stage1 VQ-VAE on tree datasets.")
    parser.add_argument("--config", default="config.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    logging_cfg = cfg.get("logging", {})
    wandb_cfg = cfg.get("wandb", {})

    train_dir = paths.get("train_dir")
    val_dir = paths.get("val_dir")
    output_dir = paths.get("output_dir", "Stage1_New/output")
    os.makedirs(output_dir, exist_ok=True)

    k = int(params.get("k", 39))
    mode = params.get("mode", "post_order")
    batch_size = int(params.get("batch_size", 1))
    batch_size_val = int(params.get("batch_size_val", 1))
    base_lr = float(params.get("base_lr", 0.0001))
    epochs = int(params.get("epochs", 1000))
    step_lr = bool(params.get("step_lr", True))
    step_size = int(params.get("step_size", 200))
    gamma = float(params.get("gamma", 0.9))
    seed = int(params.get("seed", 125))
    device = resolve_device(params.get("device", 0))
    mask_null = bool(params.get("mask_null", False))
    null_threshold = float(params.get("null_threshold", 1e-3))
    seed_all(seed)

    if k != 39:
        print("Warning: Stage1 config expects k=39 with current IntraDataset implementation.")

    num_workers = int(params.get("dataloader_num_workers", 0))
    pin_memory = bool(params.get("dataloader_pin_memory", False))
    persistent_workers = bool(params.get("dataloader_persistent_workers", False))
    prefetch_factor = params.get("dataloader_prefetch_factor", None)
    if prefetch_factor is not None:
        prefetch_factor = int(prefetch_factor)

    train_loader = build_loader(
        train_dir,
        mode,
        batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )
    val_loader = build_loader(
        val_dir,
        mode,
        batch_size_val,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
    )

    args_cfg = Args()
    args_cfg.base_lr = base_lr
    args_cfg.batch_size = batch_size
    args_cfg.batch_size_val = batch_size_val
    args_cfg.step_size = step_size
    args_cfg.gamma = gamma
    args_cfg.in_dim = int(params.get("in_dim", k))
    args_cfg.quant_loss_weight = float(params.get("quant_loss_weight", args_cfg.quant_loss_weight))

    model = VQAutoEncoder(args_cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=base_lr)
    scheduler = StepLR(optimizer, step_size=step_size, gamma=gamma) if step_lr else None

    log_every = int(logging_cfg.get("log_every", 10))
    save_every = int(logging_cfg.get("save_every", 100))
    save_best = bool(logging_cfg.get("save_best", True))
    best_model_path = logging_cfg.get("best_model_path", os.path.join(output_dir, "best-model-stage1.pth"))

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        if wandb is None:
            raise RuntimeError("wandb is enabled but not installed.")
        wandb.init(
            # entity=wandb_cfg.get("entity"),
            project=wandb_cfg.get("project"),
            mode=wandb_cfg.get("mode", "online"),
            name=wandb_cfg.get("run_name"),
            config={
                "paths": paths,
                "params": params,
                "logging": logging_cfg,
            },
        )
        if bool(wandb_cfg.get("watch", True)):
            wandb.watch(model, log="all")

    best_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_loss, rec_loss, quant_loss, perplexity = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            args_cfg.quant_loss_weight,
            mask_null=mask_null,
            null_threshold=null_threshold,
        )
        val_loss, rec_loss_val, quant_loss_val = validate(
            val_loader,
            model,
            device,
            args_cfg.quant_loss_weight,
            epoch=epoch,
            mask_null=mask_null,
            null_threshold=null_threshold,
        )

        if scheduler is not None:
            scheduler.step()

        if epoch % log_every == 0:
            print(
                f"epoch {epoch} | total {train_loss:.6f} | rec {rec_loss:.6f} | quant {quant_loss:.6f} | pp {perplexity:.4f} | val {val_loss:.6f}"
            )

        if wandb_enabled:
            wandb.log(
                {
                    "epoch": epoch,
                    "train/total_loss": train_loss,
                    "train/rec_loss": rec_loss,
                    "train/quant_loss": quant_loss,
                    "train/perplexity": perplexity,
                    "val/total_loss": val_loss,
                    "val/rec_loss": rec_loss_val,
                    "val/quant_loss": quant_loss_val,
                    "lr": optimizer.param_groups[0]["lr"],
                }
            )

        if save_best:
            best_loss = save_best_model(
                model,
                optimizer,
                epoch,
                rec_loss_val,
                best_loss,
                model_save_path=best_model_path,
            )

        if epoch % save_every == 0:
            ckpt_path = os.path.join(output_dir, f"checkpoint-epoch{epoch}.pth")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": val_loss,
                },
                ckpt_path,
            )


if __name__ == "__main__":
    main()
