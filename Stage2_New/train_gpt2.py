import argparse
import gc
import hashlib
import math
import os
import random
import sys
from array import array
from pathlib import Path

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2Config, GPT2LMHeadModel, get_scheduler
from tqdm import tqdm

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from Stage1.base.utilities import AverageMeter

try:
    import yaml
except Exception as exc:
    raise RuntimeError("PyYAML is required. Install with: pip install pyyaml") from exc

try:
    import wandb
except Exception:
    wandb = None

from funciones import save_best_model_gpt2


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg or {}


def seed_all(seed):
    random.seed(seed)
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


class TokenDataset(Dataset):
    def __init__(self, folder_path, eos_token):
        self.samples = []
        self.eos_token = eos_token
        self._load_files(folder_path)

    def _load_files(self, folder_path):
        files = os.listdir(folder_path)
        for file_name in files:
            file_path = os.path.join(folder_path, file_name)
            self.samples.append(torch.load(file_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        eos = torch.tensor([self.eos_token])
        seq = torch.cat((eos, self.samples[idx], eos))
        return seq.long()


def custom_collate(batch, pad_token_id):
    return pad_sequence(batch, batch_first=True, padding_value=pad_token_id)


def create_attention_mask(batch, pad_token_id):
    return (batch != pad_token_id).long()


def build_dataset(train_dir, eos_token, pad_token, batch_size, shuffle):
    dataset = TokenDataset(train_dir, eos_token=eos_token)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=lambda batch: custom_collate(batch, pad_token),
        shuffle=shuffle,
    )

    return dataloader


def update_token_counts(counts, batch, pad_token):
    tokens = batch[batch != pad_token].detach().cpu().view(-1)
    if tokens.numel() == 0:
        return
    counts += torch.bincount(tokens, minlength=counts.numel())


def compute_usage_perplexity(counts, exclude_ids=None):
    counts = counts.clone().float()
    if exclude_ids:
        for idx in exclude_ids:
            if 0 <= idx < counts.numel():
                counts[idx] = 0
    total = counts.sum()
    if total <= 0:
        return float("nan")
    probs = counts / total
    probs = probs[probs > 0]
    entropy = -(probs * torch.log(probs)).sum()
    return float(torch.exp(entropy))


def _strip_sequence(seq, eos_token, pad_token):
    seq = [int(x) for x in seq if int(x) != pad_token]
    if seq and seq[0] == eos_token:
        seq = seq[1:]
    if eos_token in seq:
        seq = seq[:seq.index(eos_token)]
    return seq


def _seq_hash(seq):
    if not seq:
        return None
    payload = array("H", seq).tobytes()
    return hashlib.blake2b(payload, digest_size=16).hexdigest()


def build_sequence_hashes(samples, eos_token, pad_token):
    hashes = set()
    for sample in samples:
        seq = _strip_sequence(sample.tolist(), eos_token, pad_token)
        digest = _seq_hash(seq)
        if digest is not None:
            hashes.add(digest)
    return hashes


def sample_sequences_for_mem_check(
    model,
    device,
    eos_token,
    pad_token,
    num_samples,
    max_new_tokens,
    do_sample=True,
    temperature=1.0,
    top_k=50,
    top_p=0.95,
    num_beams=1,
):
    input_ids = torch.full((num_samples, 1), eos_token, dtype=torch.long, device=device)
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        num_beams=num_beams,
        eos_token_id=eos_token,
        pad_token_id=pad_token,
    )
    return outputs


def train_one_epoch(dataloader, model, optimizer, lr_scheduler, device, pad_token, track_usage=False, vocab_size=None):
    total_loss_meter = AverageMeter()
    token_counts = None
    if track_usage:
        if vocab_size is None:
            raise ValueError("vocab_size is required when track_usage is enabled.")
        token_counts = torch.zeros(vocab_size, dtype=torch.long)
    model.train()
    for batch in tqdm(dataloader, desc="train", leave=False):
        batch = batch.to(device)
        attention_mask = create_attention_mask(batch, pad_token).to(device)
        outputs = model(batch, labels=batch, attention_mask=attention_mask)
        loss = outputs.loss

        total_loss_meter.update(loss.item(), batch.size(0))
        if track_usage:
            update_token_counts(token_counts, batch, pad_token)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()

        del outputs, loss, batch
        gc.collect()

    if track_usage:
        return total_loss_meter.avg, token_counts
    return total_loss_meter.avg


def val_one_epoch(dataloader, model, device, pad_token, track_usage=False, vocab_size=None):
    model.eval()
    total_loss_meter = AverageMeter()
    token_counts = None
    if track_usage:
        if vocab_size is None:
            raise ValueError("vocab_size is required when track_usage is enabled.")
        token_counts = torch.zeros(vocab_size, dtype=torch.long)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="val", leave=False):
            batch = batch.to(device)
            attention_mask = create_attention_mask(batch, pad_token).to(device)
            outputs = model(batch, labels=batch, attention_mask=attention_mask)
            loss = outputs.loss

            total_loss_meter.update(loss.item(), batch.size(0))
            if track_usage:
                update_token_counts(token_counts, batch, pad_token)

            del outputs, loss, batch
            gc.collect()

    model.train()
    if track_usage:
        return total_loss_meter.avg, token_counts
    return total_loss_meter.avg


def create_gpt2_model(cfg, vocab_size, max_size, pad_token):
    config = GPT2Config(
        vocab_size=vocab_size,
        n_embd=int(cfg.get("n_embd", 512)),
        n_layer=int(cfg.get("n_layer", 6)),
        n_head=int(cfg.get("n_head", 8)),
        n_positions=max_size,
        n_ctx=max_size,
        pad_token_id=pad_token,
    )
    return GPT2LMHeadModel(config)


def main():
    parser = argparse.ArgumentParser(description="Train GPT-2 on tokenized tree sequences.")
    parser.add_argument("--config", default="econfig.yaml", help="Path to YAML config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    paths = cfg.get("paths", {})
    params = cfg.get("params", {})
    model_cfg = cfg.get("model", {})
    wandb_cfg = cfg.get("wandb", {}) 

    train_dir = paths.get("train_dir")
    val_dir = paths.get("val_dir")
    output_dir = paths.get("output_dir", "Stage2_New/output")
    best_model_dir = paths.get("best_model_dir", os.path.join(output_dir, "best-gpt2"))

    if not train_dir:
        raise ValueError("paths.train_dir is required.")

    vocab_size = int(params.get("vocab_size", 258))
    max_size = int(params.get("max_size", 2258))
    pad_token = int(params.get("pad_token", 257))
    eos_token = int(params.get("eos_token", 256))
    epochs = int(params.get("epochs", 50000))
    lr = float(params.get("lr", 1e-4))
    batch_size = int(params.get("batch_size", 4))
    shuffle = bool(params.get("shuffle", False))
    warmup_steps = int(params.get("warmup_steps", 0))
    scheduler_type = params.get("scheduler", "linear")
    seed = int(params.get("seed", 12))
    device = resolve_device(params.get("device", 0))

    seed_all(seed)

    train_loader = build_dataset(train_dir, eos_token, pad_token, batch_size, shuffle)
    val_loader = None
    if val_dir:
        val_loader = build_dataset(val_dir, eos_token, pad_token, batch_size, False)

    max_train_len = max((len(seq) for seq in train_loader), default=0)
    max_val_len = max((len(seq) for seq in val_loader), default=0) if val_loader else 0
    max_len = max(max_train_len, max_val_len)
    if max_len > max_size:
        print(f"Warning: dataset max length {max_len} exceeds max_size {max_size}.")

    model = create_gpt2_model(model_cfg, vocab_size, max_size, pad_token)
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    lr_scheduler = get_scheduler(
        scheduler_type,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=len(train_loader) * epochs,
    )

    wandb_enabled = bool(wandb_cfg.get("enabled", False))
    if wandb_enabled:
        if wandb is None:
            raise RuntimeError("wandb is enabled but not installed.")
        wandb.init(
            project=wandb_cfg.get("project", "gpt2"),
            entity=wandb_cfg.get("entity"),
            mode=wandb_cfg.get("mode", "online"),
            name=wandb_cfg.get("run_name"),
            config={
                "paths": paths,
                "params": params,
                "model": model_cfg,
            },
        )

    best_loss = float("inf")
    log_every = int(params.get("log_every", 1))
    track_usage = bool(params.get("track_token_usage", False))
    usage_exclude_ids = params.get("token_usage_exclude_ids", [])
    if usage_exclude_ids is None:
        usage_exclude_ids = [] 
    usage_exclude_ids = [int(x) for x in usage_exclude_ids]

    mem_check_enabled = bool(params.get("mem_check_enabled", False))
    mem_check_every = int(params.get("mem_check_every", 1))
    mem_check_samples = int(params.get("mem_check_samples", 16))
    mem_check_max_new_tokens = int(params.get("mem_check_max_new_tokens", 512))
    mem_check_do_sample = bool(params.get("mem_check_do_sample", True))
    mem_check_temperature = float(params.get("mem_check_temperature", 1.0))
    mem_check_top_k = int(params.get("mem_check_top_k", 50))
    mem_check_top_p = float(params.get("mem_check_top_p", 0.95))
    mem_check_num_beams = int(params.get("mem_check_num_beams", 1))

    train_hashes = None
    if mem_check_enabled:
        train_hashes = build_sequence_hashes(train_loader.dataset.samples, eos_token, pad_token)

    # Training Loop
    model.train()
    for epoch in range(epochs):
        if track_usage:
            train_avg_loss, train_counts = train_one_epoch(
                train_loader,
                model,
                optimizer,
                lr_scheduler,
                device,
                pad_token,
                track_usage=True,
                vocab_size=vocab_size,
            )
            if val_loader:
                val_avg_loss, val_counts = val_one_epoch(
                    val_loader,
                    model,
                    device,
                    pad_token,
                    track_usage=True,
                    vocab_size=vocab_size,
                )
            train_usage_ppl = compute_usage_perplexity(train_counts, usage_exclude_ids)
            val_usage_ppl = compute_usage_perplexity(val_counts, usage_exclude_ids) if val_loader else None
        else:
            train_avg_loss = train_one_epoch(train_loader, model, optimizer, lr_scheduler, device, pad_token)
            val_avg_loss = val_one_epoch(val_loader, model, device, pad_token) if val_loader else None
            train_usage_ppl = None
            val_usage_ppl = None

        try:
            train_ppl = math.exp(train_avg_loss)
        except OverflowError:
            train_ppl = float("inf")
        if val_avg_loss is not None:
            try:
                val_ppl = math.exp(val_avg_loss)
            except OverflowError:
                val_ppl = float("inf")
        else:
            val_ppl = None

        mem_exact_match_rate = None
        mem_unique_rate = None
        if mem_check_enabled and (epoch % mem_check_every == 0):
            model.eval()
            with torch.no_grad():
                gen = sample_sequences_for_mem_check(
                    model,
                    device,
                    eos_token,
                    pad_token,
                    mem_check_samples,
                    mem_check_max_new_tokens,
                    do_sample=mem_check_do_sample,
                    temperature=mem_check_temperature,
                    top_k=mem_check_top_k,
                    top_p=mem_check_top_p,
                    num_beams=mem_check_num_beams,
                )
            model.train()
            gen_hashes = []
            for seq in gen:
                stripped = _strip_sequence(seq.tolist(), eos_token, pad_token)
                digest = _seq_hash(stripped)
                if digest is not None:
                    gen_hashes.append(digest)
            if gen_hashes:
                hits = sum(1 for h in gen_hashes if h in train_hashes)
                mem_exact_match_rate = hits / len(gen_hashes)
                mem_unique_rate = len(set(gen_hashes)) / len(gen_hashes)

        if epoch % log_every == 0:
            msg = f"Epoch {epoch} | Train Avg Loss: {train_avg_loss} |  Train PPL: {train_ppl}"
            if val_avg_loss is not None:
                msg += f" | Val Avg Loss: {val_avg_loss} | Val PPL: {val_ppl}"
            if track_usage:
                msg += f" | Train Usage PPL: {train_usage_ppl}"
                if val_usage_ppl is not None:
                    msg += f" | Val Usage PPL: {val_usage_ppl}"
            if mem_exact_match_rate is not None:
                msg += (
                    f" | Mem ExactMatch: {mem_exact_match_rate:.3f}"
                    f" | Mem Unique: {mem_unique_rate:.3f}"
                )
            print(msg)

        if wandb_enabled:
            payload = {
                "epoch": epoch,
                "avg_loss": train_avg_loss,
                "train_perplexity": train_ppl,
            }
            if val_avg_loss is not None:
                payload.update(
                    {
                        "val_avg_loss": val_avg_loss,
                        "val_perplexity": val_ppl,
                    }
                )
            if track_usage:
                payload.update(
                    {
                        "train_usage_perplexity": train_usage_ppl,
                    }
                )
                if val_usage_ppl is not None:
                    payload.update(
                        {
                            "val_usage_perplexity": val_usage_ppl,
                        }
                    )
            if mem_exact_match_rate is not None:
                payload.update(
                    {
                        "mem_exact_match_rate": mem_exact_match_rate,
                        "mem_unique_rate": mem_unique_rate,
                    }
                )
            wandb.log(payload)

        os.makedirs(output_dir, exist_ok=True)
        best_loss = save_best_model_gpt2(
            model, optimizer, epoch, train_avg_loss, best_loss, best_model_dir
        )


if __name__ == "__main__":
    main()
