#!/usr/bin/env python3
"""Train a lightweight Boxer-Residual-LIFT head."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from tqdm import tqdm

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from tools.lifthead_common import ResidualLiftHead  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Boxer-Residual-LIFT.")
    parser.add_argument("--train_pth", required=True)
    parser.add_argument("--val_pth", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--category_embed_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--force_cpu", action="store_true")
    parser.add_argument("--loss_center", type=float, default=2.0)
    parser.add_argument("--loss_dims", type=float, default=1.0)
    parser.add_argument("--loss_yaw", type=float, default=0.35)
    return parser.parse_args()


def load_data(path: str) -> Dict:
    data = torch.load(path, map_location="cpu")
    required = ["features", "category_indices", "targets", "weights"]
    for key in required:
        if key not in data:
            raise ValueError(f"{path} missing {key}")
    return data


def make_dataset(data: Dict, mean: torch.Tensor, std: torch.Tensor) -> TensorDataset:
    features = (data["features"].float() - mean) / std
    category_indices = data["category_indices"].long()
    targets = data["targets"].float()
    weights = data["weights"].float().clamp(min=0.05, max=2.0)
    return TensorDataset(features, category_indices, targets, weights)


def weighted_mean(loss: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    while weights.ndim < loss.ndim:
        weights = weights.unsqueeze(-1)
    return (loss * weights).sum() / weights.sum().clamp(min=1e-6)


def compute_loss(pred: torch.Tensor, target: torch.Tensor, weights: torch.Tensor, args: argparse.Namespace) -> Tuple[torch.Tensor, Dict[str, float]]:
    center_loss = F.smooth_l1_loss(pred[:, :3], target[:, :3], reduction="none").mean(dim=1)
    dims_loss = F.smooth_l1_loss(pred[:, 3:6], target[:, 3:6], reduction="none").mean(dim=1)
    yaw_loss = F.mse_loss(pred[:, 6:8], target[:, 6:8], reduction="none").mean(dim=1)
    total = (
        args.loss_center * weighted_mean(center_loss, weights)
        + args.loss_dims * weighted_mean(dims_loss, weights)
        + args.loss_yaw * weighted_mean(yaw_loss, weights)
    )
    metrics = {
        "loss": float(total.detach().cpu()),
        "center": float(weighted_mean(center_loss, weights).detach().cpu()),
        "dims": float(weighted_mean(dims_loss, weights).detach().cpu()),
        "yaw": float(weighted_mean(yaw_loss, weights).detach().cpu()),
    }
    return total, metrics


def run_epoch(
    model: ResidualLiftHead,
    loader: DataLoader,
    device: torch.device,
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer | None = None,
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    totals = {"loss": 0.0, "center": 0.0, "dims": 0.0, "yaw": 0.0, "n": 0}
    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for features, cats, targets, weights in loader:
            features = features.to(device)
            cats = cats.to(device)
            targets = targets.to(device)
            weights = weights.to(device)
            pred = model(features, cats)
            loss, metrics = compute_loss(pred, targets, weights, args)
            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
            bsz = int(features.shape[0])
            for key in ("loss", "center", "dims", "yaw"):
                totals[key] += metrics[key] * bsz
            totals["n"] += bsz
    n = max(int(totals.pop("n")), 1)
    return {key: value / n for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available() and not args.force_cpu:
        torch.cuda.set_device(args.gpu)
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    train_data = load_data(args.train_pth)
    feature_mean = train_data["features"].float().mean(dim=0)
    feature_std = train_data["features"].float().std(dim=0).clamp(min=1e-4)
    train_full = make_dataset(train_data, feature_mean, feature_std)

    if args.val_pth:
        val_data = load_data(args.val_pth)
        train_dataset = train_full
        val_dataset = make_dataset(val_data, feature_mean, feature_std)
    else:
        val_len = int(round(len(train_full) * args.val_ratio))
        train_len = len(train_full) - val_len
        train_dataset, val_dataset = random_split(
            train_full,
            [train_len, val_len],
            generator=torch.Generator().manual_seed(args.seed),
        )
        val_data = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=(device.type == "cuda"))
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=(device.type == "cuda"))

    num_categories = int(train_data["category_indices"].max().item()) + 1
    if val_data is not None:
        num_categories = max(num_categories, int(val_data["category_indices"].max().item()) + 1)
    model = ResidualLiftHead(
        feature_dim=train_data["features"].shape[1],
        num_categories=num_categories,
        category_embed_dim=args.category_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    os.makedirs(args.output_dir, exist_ok=True)
    best_val = float("inf")
    history = []
    for epoch in tqdm(range(1, args.epochs + 1), desc="Training LiftHead"):
        train_metrics = run_epoch(model, train_loader, device, args, optimizer)
        val_metrics = run_epoch(model, val_loader, device, args, None)
        scheduler.step()
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics, "lr": scheduler.get_last_lr()[0]}
        history.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(
                f"epoch={epoch:03d} train={train_metrics['loss']:.5f} "
                f"val={val_metrics['loss']:.5f} center={val_metrics['center']:.5f} "
                f"dims={val_metrics['dims']:.5f} yaw={val_metrics['yaw']:.5f}"
            )
        ckpt = {
            "model": model.state_dict(),
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "feature_names": train_data.get("feature_names"),
            "roi_feature_config": train_data.get("roi_feature_config", {"mode": "none"}),
            "depth_feature_config": train_data.get("depth_feature_config", {"mode": "none"}),
            "roi_feature_cache_config": train_data.get("roi_feature_cache_config", {}),
            "roi_feature_cache_names": train_data.get("roi_feature_cache_names", []),
            "category_id_to_index": train_data.get("category_id_to_index"),
            "num_categories": num_categories,
            "model_args": {
                "feature_dim": int(train_data["features"].shape[1]),
                "num_categories": int(num_categories),
                "category_embed_dim": args.category_embed_dim,
                "hidden_dim": args.hidden_dim,
                "num_layers": args.num_layers,
                "dropout": args.dropout,
            },
            "train_pth": os.path.abspath(args.train_pth),
            "val_pth": os.path.abspath(args.val_pth) if args.val_pth else None,
            "epoch": epoch,
            "val_loss": val_metrics["loss"],
        }
        torch.save(ckpt, os.path.join(args.output_dir, "last.pth"))
        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(ckpt, os.path.join(args.output_dir, "best.pth"))

    with open(os.path.join(args.output_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"Best validation loss: {best_val:.6f}")
    print(f"Wrote checkpoints to: {args.output_dir}")


if __name__ == "__main__":
    main()
