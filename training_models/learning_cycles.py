import glob
import json
import os
import random
import time
import uuid

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import matplotlib.pyplot as plt


def make_run_id(model_name):
    return f"{time.strftime('%Y%m%dT%H%M%S')}_{model_name}_{uuid.uuid4().hex[:8]}"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class IndexedDataset(Dataset):
    """Wraps a dataset so each item also yields its own stable index, which
    survives DataLoader shuffling. Without this, a sample's identity across
    epochs can only be inferred from its position in a shuffled batch, which
    changes every epoch.
    """

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data, label = self.dataset[idx]
        return data, label, idx


def per_sample_loss(logits, labels):
    return F.cross_entropy(logits, labels, reduction="none")


class MetricsRecorder:
    """Records training-dynamics telemetry for one run across four linked
    tables (joined on run_id, and sample_id where applicable):

      - "{save_path}_identifiers.parquet": one row per sample - sample_id,
        ground-truth label. Written once, before training starts.
      - "{save_path}_samples.parquet": one row per (sample_id, epoch) - full
        logit vector, per-sample loss, predicted class, correctness, and
        whether the sample was kept (used in the backward pass) that epoch.
        Internally this is written as one Parquet file per epoch
        ("{save_path}_samples_epoch{N}.parquet", finalized - writer closed -
        as soon as that epoch's batches are done) rather than one writer
        spanning the whole run: Parquet only becomes readable once its
        footer is written on close(), so a single run-long writer means a
        crash at epoch 190 corrupts all 190 epochs' worth of sample data,
        not just the in-flight one. merge_epoch_samples() concatenates the
        per-epoch parts into the canonical path once the run finishes
        successfully.
      - "{save_path}_epochs.parquet": one row per epoch - loss/accuracy
        (train & val), their gap and epoch-over-epoch velocity, learning
        rate, epoch fraction of budget, samples used, and distributional
        summaries (mean/median/std of loss and confidence) across the
        dataset that epoch. NOTE on samples_used vs samples_total: for the
        batch-masking DBPD variants (critical_periods_dbpd phase 2,
        delayed_dbpd, soft_dbpd, combined), every sample in the loader gets
        a forward pass each epoch (used to decide the hard/easy mask)
        before any masking happens - samples_total counts that full forward
        pass, not a nominal denominator. samples_used counts only the
        (usually smaller) subset that got a backward pass. Time spent is
        dominated by the samples_total forward pass, not samples_used, so
        samples_used / samples_total is not a proxy for wall-clock savings
        - see time_seconds for that.
      - "{save_path}_outcomes.parquet": one row per sample - the final
        trained model's correctness/confidence/loss on that sample. Written
        once, after training completes.
      - "{save_path}_run_metadata.json": run-level config (architecture,
        dataset, optimizer, seed, epochs, batch size, ...), not tabular.

    Also renders "{save_path}_dynamics.png": loss/accuracy/LR/velocity/
    distributional-summary curves over epochs, once training finishes.
    """

    def __init__(self, model_name, save_path, run_id, seed, dataset, architecture,
                 optimizer_name, learning_rate, total_epochs, batch_size,
                 threshold=None, start_revision=None, mode=None, **extra_metadata):
        self.model_name = model_name
        self.run_id = run_id
        self.seed = seed
        self.total_epochs = total_epochs

        base = save_path.rstrip("/\\")
        out_dir = os.path.dirname(base)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        self.base = base

        self.identifiers_path = f"{base}_identifiers.parquet"
        self.sample_path = f"{base}_samples.parquet"
        self.epoch_path = f"{base}_epochs.parquet"
        self.outcomes_path = f"{base}_outcomes.parquet"
        self.metadata_path = f"{base}_run_metadata.json"
        self.dynamics_plot_path = f"{base}_dynamics.png"

        metadata = {
            "run_id": run_id,
            "model_name": model_name,
            "architecture": architecture,
            "dataset": dataset,
            "optimizer": optimizer_name,
            "learning_rate": learning_rate,
            "total_epochs": total_epochs,
            "batch_size": batch_size,
            "threshold": threshold,
            "start_revision": start_revision,
            "mode": mode,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        metadata.update(extra_metadata)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self._samples_writer = None
        self._epoch_rows = []
        self._epoch_loss_buffer = []
        self._epoch_conf_buffer = []
        self._prev = {"train_loss": None, "val_loss": None, "train_accuracy": None, "val_accuracy": None}

    def update_metadata(self, **kwargs):
        with open(self.metadata_path) as f:
            metadata = json.load(f)
        metadata.update(kwargs)
        with open(self.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

    def log_identifiers(self, sample_ids, labels):
        df = pd.DataFrame({
            "run_id": self.run_id,
            "sample_id": list(sample_ids),
            "label": list(labels),
            "seed": self.seed,
        })
        df.to_parquet(self.identifiers_path, index=False)

    def log_batch(self, epoch, sample_ids, labels, preds, logits, losses, probs_true_class, kept_mask, phase=None):
        sample_ids = list(sample_ids)
        labels_list = labels.detach().cpu().tolist()
        preds_list = preds.detach().cpu().tolist()
        logits_list = logits.detach().cpu().tolist()
        losses_list = losses.detach().cpu().tolist()
        probs_list = probs_true_class.detach().cpu().tolist()
        kept_list = [bool(k) for k in kept_mask.detach().cpu().tolist()]

        table = pa.table({
            "run_id": [self.run_id] * len(sample_ids),
            "sample_id": sample_ids,
            "epoch": [epoch] * len(sample_ids),
            "phase": [phase] * len(sample_ids),
            "logits": logits_list,
            "loss": losses_list,
            "predicted_class": preds_list,
            "correct": [p == l for p, l in zip(preds_list, labels_list)],
            "true_class_probability": probs_list,
            "kept": kept_list,
        })
        if self._samples_writer is None:
            self._samples_writer = pq.ParquetWriter(f"{self.base}_samples_epoch{epoch}.parquet", table.schema)
        self._samples_writer.write_table(table)

        self._epoch_loss_buffer.extend(losses_list)
        self._epoch_conf_buffer.extend(probs_list)

    def finalize_epoch_samples(self):
        """Closes (and writes the footer for) the current epoch's samples
        writer. Call once a given epoch's batches are all logged, so that
        epoch's data is durably readable on disk before moving on - a crash
        during a later epoch then only costs that later epoch's samples.
        """
        if self._samples_writer is not None:
            self._samples_writer.close()
            self._samples_writer = None

    def export_state(self):
        """Snapshot of in-memory accumulators, for embedding in a training
        checkpoint so a resumed process's recorder can pick up where the
        crashed one left off (see dbpd_mechanisms.py's checkpoint/resume).
        """
        return {"epoch_rows": list(self._epoch_rows), "prev": dict(self._prev)}

    def import_state(self, state):
        self._epoch_rows = list(state["epoch_rows"])
        self._prev = dict(state["prev"])

    def log_epoch(self, epoch, learning_rate, train_loss, train_accuracy, val_loss,
                  val_accuracy, samples_used, samples_total, time_seconds, threshold=None,
                  phase=None, gradient_confusion=None, hard_fraction=None):
        loss_arr = np.array(self._epoch_loss_buffer) if self._epoch_loss_buffer else np.array([np.nan])
        conf_arr = np.array(self._epoch_conf_buffer) if self._epoch_conf_buffer else np.array([np.nan])

        def velocity(key, current):
            prev = self._prev[key]
            return float("nan") if prev is None else current - prev

        row = {
            "run_id": self.run_id,
            "epoch": epoch,
            "phase": phase,
            "epoch_fraction": (epoch + 1) / self.total_epochs,
            "threshold": threshold,
            "gradient_confusion": gradient_confusion,
            "hard_fraction": hard_fraction,
            "learning_rate": learning_rate,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "loss_gap": val_loss - train_loss,
            "train_loss_velocity": velocity("train_loss", train_loss),
            "val_loss_velocity": velocity("val_loss", val_loss),
            "train_accuracy": train_accuracy,
            "val_accuracy": val_accuracy,
            "accuracy_gap": train_accuracy - val_accuracy,
            "train_accuracy_velocity": velocity("train_accuracy", train_accuracy),
            "val_accuracy_velocity": velocity("val_accuracy", val_accuracy),
            "samples_used": samples_used,
            "samples_total": samples_total,
            "time_seconds": time_seconds,
            "mean_loss": float(np.mean(loss_arr)),
            "median_loss": float(np.median(loss_arr)),
            "std_loss": float(np.std(loss_arr)),
            "mean_confidence": float(np.mean(conf_arr)),
            "median_confidence": float(np.median(conf_arr)),
            "std_confidence": float(np.std(conf_arr)),
        }
        self._epoch_rows.append(row)

        self._prev["train_loss"] = train_loss
        self._prev["val_loss"] = val_loss
        self._prev["train_accuracy"] = train_accuracy
        self._prev["val_accuracy"] = val_accuracy
        self._epoch_loss_buffer = []
        self._epoch_conf_buffer = []

        # Flushed immediately (not just at close()) so the epochs table is
        # durable up through the last completed epoch even if the process
        # crashes before finishing the run.
        pd.DataFrame(self._epoch_rows).to_parquet(self.epoch_path, index=False)

    def log_outcomes(self, sample_ids, labels, preds, logits, losses, probs_true_class):
        sample_ids = list(sample_ids)
        labels_list = labels.detach().cpu().tolist()
        preds_list = preds.detach().cpu().tolist()
        df = pd.DataFrame({
            "run_id": self.run_id,
            "sample_id": sample_ids,
            "final_correct": [p == l for p, l in zip(preds_list, labels_list)],
            "final_predicted_class": preds_list,
            "final_confidence": probs_true_class.detach().cpu().tolist(),
            "final_loss": losses.detach().cpu().tolist(),
            "final_logits": logits.detach().cpu().tolist(),
        })
        df.to_parquet(self.outcomes_path, index=False)

    def _plot_dynamics(self):
        if not self._epoch_rows:
            return
        df = pd.DataFrame(self._epoch_rows)

        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle(f"Training Dynamics: {self.model_name}")

        ax = axes[0, 0]
        ax.plot(df["epoch"], df["train_loss"], label="train_loss", marker="o")
        ax.plot(df["epoch"], df["val_loss"], label="val_loss", marker="o")
        ax.fill_between(df["epoch"], df["train_loss"], df["val_loss"], alpha=0.15, label="gap")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Loss"); ax.set_title("Train/Val Loss"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 1]
        ax.plot(df["epoch"], df["train_accuracy"], label="train_accuracy", marker="o")
        ax.plot(df["epoch"], df["val_accuracy"], label="val_accuracy", marker="o")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Accuracy"); ax.set_title("Train/Val Accuracy"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[0, 2]
        ax.plot(df["epoch"], df["learning_rate"], marker="o", color="tab:green")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate Schedule"); ax.grid(alpha=0.3)

        ax = axes[1, 0]
        ax.plot(df["epoch"], df["train_accuracy_velocity"], label="train_accuracy_velocity", marker="o")
        ax.plot(df["epoch"], df["val_accuracy_velocity"], label="val_accuracy_velocity", marker="o")
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.set_xlabel("Epoch"); ax.set_ylabel("Δ Accuracy"); ax.set_title("Accuracy Velocity"); ax.legend(); ax.grid(alpha=0.3)

        ax = axes[1, 1]
        ax.plot(df["epoch"], df["mean_confidence"], label="mean_confidence", marker="o", color="tab:purple")
        ax.fill_between(df["epoch"], df["mean_confidence"] - df["std_confidence"],
                         df["mean_confidence"] + df["std_confidence"], alpha=0.2, color="tab:purple")
        ax.set_xlabel("Epoch"); ax.set_ylabel("True-class probability"); ax.set_title("Confidence (mean ± std)"); ax.grid(alpha=0.3)

        ax = axes[1, 2]
        ax.plot(df["epoch"], df["mean_loss"], label="mean_loss", marker="o", color="tab:red")
        ax.fill_between(df["epoch"], df["mean_loss"] - df["std_loss"],
                         df["mean_loss"] + df["std_loss"], alpha=0.2, color="tab:red")
        ax.set_xlabel("Epoch"); ax.set_ylabel("Per-sample loss"); ax.set_title("Loss (mean ± std)"); ax.grid(alpha=0.3)

        plt.tight_layout()
        plt.savefig(self.dynamics_plot_path)
        plt.close(fig)

    def merge_epoch_samples(self):
        """Concatenates the per-epoch "{base}_samples_epoch{N}.parquet"
        parts (see log_batch/finalize_epoch_samples) into the canonical
        "{base}_samples.parquet", copying row groups directly instead of
        materializing everything in memory at once (this table can reach
        millions of rows over a full run). Call only once the run has
        finished successfully - crashed/incomplete runs leave the parts as
        they are, individually readable, for inspection.
        """
        self.finalize_epoch_samples()
        parts = sorted(
            glob.glob(f"{self.base}_samples_epoch*.parquet"),
            key=lambda p: int(os.path.basename(p).rsplit("epoch", 1)[1].split(".")[0]),
        )
        if not parts:
            return
        tmp_path = f"{self.sample_path}.tmp"
        writer = None
        for part in parts:
            # Opened via an explicit file handle (rather than handing
            # pq.ParquetFile a bare path) so the handle is guaranteed closed
            # by the `with` block - on Windows, os.remove() below fails with
            # a file-in-use error if pyarrow is still holding it open.
            with open(part, "rb") as fh:
                pf = pq.ParquetFile(fh)
                for record_batch in pf.iter_batches():
                    batch_table = pa.Table.from_batches([record_batch])
                    if writer is None:
                        writer = pq.ParquetWriter(tmp_path, batch_table.schema)
                    writer.write_table(batch_table)
        if writer is not None:
            writer.close()
            os.replace(tmp_path, self.sample_path)
        for part in parts:
            # Best-effort: the canonical file above is already correct at
            # this point, so a stray leftover part (e.g. a lingering handle
            # on Windows) is cosmetic, not a reason to fail the whole run.
            try:
                os.remove(part)
            except OSError:
                pass

    def close(self):
        if self._samples_writer is not None:
            self._samples_writer.close()
        pd.DataFrame(self._epoch_rows).to_parquet(self.epoch_path, index=False)
        self._plot_dynamics()

        print(f"Identifiers saved to '{self.identifiers_path}'")
        print(f"Sample-level metrics saved to '{self.sample_path}'")
        print(f"Epoch-level metrics saved to '{self.epoch_path}'")
        print(f"Run metadata saved to '{self.metadata_path}'")
        print(f"Training dynamics plot saved to '{self.dynamics_plot_path}'")