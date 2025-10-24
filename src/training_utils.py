"""Utilities for training captioning models with logging and early stopping."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import torch


@dataclass
class TrainingMetrics:
    """Stores per-epoch training statistics."""

    epochs: List[int]
    losses: List[float]
    bleus: List[float]
    times: List[float]
    best_epoch: Optional[int] = None
    best_bleu: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return metrics as a JSON-serializable dictionary."""
        return {
            "epochs": self.epochs,
            "losses": self.losses,
            "bleus": self.bleus,
            "times": self.times,
            "best_epoch": self.best_epoch,
            "best_bleu": self.best_bleu,
            "total_training_time": sum(self.times),
        }


def train_with_early_stopping(
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    train_loader: Iterable,
    dev_loader: Iterable,
    criterion: Callable[..., torch.Tensor],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    vocab_size: int,
    *,
    num_epochs: int = 500,
    eval_interval: int = 10,
    patience: int = 5,
    min_delta: float = 0.0,
    checkpoint_dir: Path | str = "checkpoints",
    evaluate_bleu_fn: Optional[Callable[..., float]] = None,
    evaluate_bleu_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> TrainingMetrics:
    """Train encoder/decoder modules with periodic evaluation and early stopping.

    Args:
        encoder: Feature extractor module.
        decoder: Caption decoder module.
        train_loader: Dataloader for training samples.
        dev_loader: Dataloader for validation samples.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Target device for tensors.
        vocab_size: Vocabulary size for logits reshaping.
        num_epochs: Maximum number of training epochs.
        eval_interval: Evaluate BLEU and save checkpoints every N epochs.
        patience: Stop after this many evaluations without BLEU improvement.
        min_delta: Minimum BLEU improvement required to reset the patience counter.
        checkpoint_dir: Directory where checkpoints are persisted.
        evaluate_bleu_fn: Callable used to compute BLEU; must accept `dev_loader`
            and any keyword arguments provided through `evaluate_bleu_kwargs`.
        evaluate_bleu_kwargs: Extra keyword arguments forwarded to `evaluate_bleu_fn`.
        verbose: If True, prints progress information.

    Returns:
        TrainingMetrics populated with loss, BLEU, timing data, and best epoch.
    """
    if evaluate_bleu_fn is None:
        raise ValueError("evaluate_bleu_fn must be provided to compute BLEU scores.")

    checkpoint_path = Path(checkpoint_dir)
    checkpoint_path.mkdir(parents=True, exist_ok=True)

    epochs: List[int] = []
    losses: List[float] = []
    bleus: List[float] = []
    times: List[float] = []

    best_bleu = float("-inf")
    best_epoch = None
    epochs_since_best = 0

    bleu_kwargs = evaluate_bleu_kwargs or {}

    for epoch in range(1, num_epochs + 1):
        encoder.train()
        decoder.train()

        start_time = time.perf_counter()
        running_loss = 0.0
        batches_seen = 0

        for imgs, caps in train_loader:
            imgs = imgs.to(device)
            caps = caps.to(device)

            feats = encoder(imgs)
            outputs = decoder(feats, caps[:, :-1])

            loss = criterion(outputs.reshape(-1, vocab_size), caps.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            batches_seen += 1

        epoch_time = time.perf_counter() - start_time
        avg_loss = running_loss / max(batches_seen, 1)

        should_evaluate = epoch == 1 or epoch % eval_interval == 0 or epoch == num_epochs

        if should_evaluate:
            decoder.eval()
            encoder.eval()

            with torch.no_grad():
                bleu = evaluate_bleu_fn(dev_loader, **bleu_kwargs)

            epochs.append(epoch)
            losses.append(avg_loss)
            bleus.append(bleu)
            times.append(epoch_time)

            torch.save(
                {
                    "epoch": epoch,
                    "encoder_state": encoder.state_dict(),
                    "decoder_state": decoder.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "loss": avg_loss,
                    "bleu": bleu,
                },
                checkpoint_path / f"checkpoint_epoch_{epoch}.pth",
            )

            if verbose:
                print(
                    f"Epoch {epoch}: loss={avg_loss:.4f}, bleu={bleu:.4f}, "
                    f"time={epoch_time:.2f}s"
                )

            if bleu > best_bleu + min_delta:
                best_bleu = bleu
                best_epoch = epoch
                epochs_since_best = 0
            else:
                epochs_since_best += 1
                if patience and epochs_since_best >= patience:
                    if verbose:
                        print(
                            f"Early stopping triggered at epoch {epoch} "
                            f"(best BLEU {best_bleu:.4f} at epoch {best_epoch})."
                        )
                    break

    metrics = TrainingMetrics(
        epochs=epochs,
        losses=losses,
        bleus=bleus,
        times=times,
        best_epoch=best_epoch,
        best_bleu=best_bleu if best_epoch is not None else None,
    )

    if verbose:
        if metrics.best_epoch is not None:
            print(
                f"Best epoch based on BLEU: {metrics.best_epoch} "
                f"with BLEU {metrics.best_bleu:.4f}"
            )
        print(f"Total tracked training time: {sum(metrics.times):.2f}s")

    return metrics


def plot_training_metrics(
    metrics: TrainingMetrics,
    *,
    save_path: Optional[Path | str] = None,
) -> None:
    """Plot loss and BLEU trends using Matplotlib."""
    import matplotlib.pyplot as plt

    if not metrics.epochs:
        raise ValueError("No metrics recorded; cannot plot empty results.")

    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(metrics.epochs, metrics.losses, marker="o")
    plt.title("Loss per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.subplot(1, 2, 2)
    plt.plot(metrics.epochs, metrics.bleus, marker="o", color="orange")
    plt.title("BLEU per Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("BLEU Score")

    plt.tight_layout()

    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
    else:
        plt.show()
