import itertools
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.v2 as T2
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils import log_memory, plot_accuracy_time_multi, plot_accuracy_time_multi_test
from learning_cycles import MetricsRecorder, IndexedDataset, per_sample_loss, make_run_id


def flatten_gradients(model):
    grads = [p.grad.detach().reshape(-1) for p in model.parameters() if p.grad is not None]
    return torch.cat(grads)


def sample_probe_batches(loader, num_batches):
    """Grabs a handful of fresh batches from a DataLoader (a new iterator each
    call, so this never disturbs the loader used for actual training).
    """
    return [(inputs, labels) for inputs, labels, _ in itertools.islice(iter(loader), num_batches)]


def compute_gradient_confusion(model, criterion, probe_batches, device, mode="max"):
    """Gradient Confusion metric (Achille et al., critical learning periods):
    the [mode] pairwise cosine similarity between the loss gradients of
    different batches. A PyTorch port of the Keras/TF GradientConfusion
    callback in critical-periods-main/utils/custom_callbacks.py, vectorized
    with a single matmul instead of the original's nested-loop + scipy calls.

    Deviation from the original: the source callback recomputes gradients
    over EVERY batch of the full training set each time it runs (~1500+
    batches for CIFAR-10 at batch_size=32), an O(n_batches^2) cosine-
    similarity cost that isn't practical here. This probes a small sample of
    batches instead (see sample_probe_batches) - same signal, tractable cost.
    """
    model.eval()
    grad_vectors = []
    for inputs, labels in probe_batches:
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        grad_vectors.append(flatten_gradients(model))
    model.zero_grad(set_to_none=True)
    model.train()

    grad_matrix = torch.stack(grad_vectors)
    normed = torch.nn.functional.normalize(grad_matrix, dim=1)
    cosine_sim = normed @ normed.T
    n = cosine_sim.size(0)
    off_diagonal = cosine_sim[~torch.eye(n, dtype=torch.bool, device=cosine_sim.device)]

    if mode == "max":
        return off_diagonal.max().item()
    elif mode == "min":
        return off_diagonal.min().item()
    return off_diagonal.mean().item()


def is_outlier(past_values, new_value, threshold=1.5, mode="low"):
    """IQR-based outlier check - direct port of critical-periods-main's
    utils/custom_functions.py:is_outlier.
    """
    past_series = pd.Series(past_values)
    q1 = past_series.quantile(0.25)
    q3 = past_series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - threshold * iqr
    upper_bound = q3 + threshold * iqr

    if mode == "low":
        return new_value < lower_bound
    elif mode == "high":
        return new_value > upper_bound
    return new_value < lower_bound or new_value > upper_bound


def build_phase1_augmentation(sample_shape):
    """Real-time augmentation for the critical-period phase, matching the
    spirit of the original's Keras ImageDataGenerator (rotation_range=15,
    width/height_shift_range=0.1, horizontal_flip=True). Operates directly
    on a batch tensor via torchvision.transforms.v2.
    """
    height, width = sample_shape[-2], sample_shape[-1]
    return T2.Compose([
        T2.RandomCrop((height, width), padding=4, padding_mode="reflect"),
        T2.RandomHorizontalFlip(p=0.5),
        T2.RandomRotation(degrees=15),
    ])


def _evaluate(model, test_loader, criterion, device):
    model.eval()
    correct, total, test_loss = 0, 0, 0.0
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating"):
            inputs = batch[0].to(device)
            labels = batch[1].to(device)
            outputs = model(inputs)
            test_loss += criterion(outputs, labels).item()
            predictions = torch.argmax(outputs, dim=-1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total, test_loss / len(test_loader)


def train_critical_periods_dbpd(model_name, model, train_loader, test_loader, device, epochs, save_path,
                                 threshold, task, cls_num_list,
                                 critical_window=15, gc_check_interval=5, gc_num_batches=10,
                                 gc_metric_frequency=1, gc_outlier_threshold=1.5,
                                 final_revision_epochs=1,
                                 run_id=None, seed=None, dataset_name=None):
    """Two-phase training strategy:

      Phase 1 (critical period): trains on every sample with real-time data
      augmentation, exactly like standard training, while monitoring the
      Gradient Confusion metric each epoch. Once that metric drops as a
      statistical low outlier relative to its own recent history (or
      critical_window epochs elapse, whichever comes first), the critical
      period is considered over.

      Phase 2 (DBPD): continues training the SAME model/optimizer (no reset)
      for the remaining epoch budget, now with augmentation OFF and
      Difficulty-Based Progressive Dropout active - each batch is filtered to
      samples whose true-class probability is below `threshold` before the
      backward pass, exactly as in selective_gradient.TrainRevision.train_with_revision.
      The final `final_revision_epochs` epochs of that budget switch to
      training on the full, unfiltered dataset - mirroring the "revision"
      tail of train_with_revision (there, everything from epoch
      `start_revision` onward is unfiltered).

    Ported from critical-periods-main (Keras/TensorFlow) - see that module's
    apply_cp.py and utils/custom_callbacks.py:GradientConfusion for the
    original. This is a from-scratch PyTorch re-implementation of the same
    mechanism, not a direct code copy (the source's model/data pipeline is
    incompatible with this codebase's DataLoader-based training loops).
    """
    model.to(device)
    if task == "classification":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    run_id = run_id if run_id else make_run_id(model_name)
    indexed_train_loader = DataLoader(IndexedDataset(train_loader.dataset), batch_size=train_loader.batch_size, shuffle=True)
    recorder = MetricsRecorder(
        model_name, save_path, run_id=run_id, seed=seed, dataset=dataset_name, architecture=model_name,
        optimizer_name="AdamW", learning_rate=3e-4, total_epochs=epochs, batch_size=train_loader.batch_size,
        threshold=threshold, start_revision=None, mode="critical_periods_dbpd",
        critical_window=critical_window, gc_check_interval=gc_check_interval,
        gc_num_batches=gc_num_batches, gc_metric_frequency=gc_metric_frequency,
        gc_outlier_threshold=gc_outlier_threshold, final_revision_epochs=final_revision_epochs,
    )

    # sample_inputs, _, _ = next(iter(indexed_train_loader))
    # augment = build_phase1_augmentation(sample_inputs.shape)

    epoch_losses, epoch_accuracies = [], []
    epoch_test_accuracies, epoch_test_losses = [], []
    time_per_epoch = []
    samples_used_per_epoch = []
    gc_history = []
    num_step = 0
    start_time = time.time()
    epoch0_sample_ids, epoch0_labels = [], []
    critical_period_end_epoch = None

    # ---------------- Phase 1: augmented training, monitored by Gradient Confusion ----------------
    phase1_epochs_run = min(critical_window, epochs)
    for epoch in range(phase1_epochs_run):
        samples_used = 0
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_start_time = time.time()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader),
                             desc=f"Phase 1 (critical period) Epoch {epoch + 1}")
        for batch_idx, (inputs, labels, sample_ids) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            # inputs = augment(inputs)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            num_step += len(outputs)
            samples_used += len(outputs)
            running_loss += loss.item()

            with torch.no_grad():
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)
                kept_mask = torch.ones_like(labels, dtype=torch.bool)  # phase 1 trains on every sample
                recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses,
                                    correct_class, kept_mask, phase="phase1")
                if epoch == 0:
                    epoch0_sample_ids.extend(sample_ids.tolist())
                    epoch0_labels.extend(labels.detach().cpu().tolist())
            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)
            progress_bar.set_postfix({"Loss": loss.item()})

        if epoch == 0:
            recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

        epoch_loss = running_loss / len(indexed_train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        time_per_epoch.append(time.time() - epoch_start_time)
        print(f"[Phase 1] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        accuracy, val_loss = _evaluate(model, test_loader, criterion, device)
        print(f"[Phase 1] Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step()
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)
        samples_used_per_epoch.append(samples_used)

        gc_value = None
        if (epoch + 1) % gc_metric_frequency == 0:
            probe_batches = sample_probe_batches(indexed_train_loader, gc_num_batches)
            gc_value = compute_gradient_confusion(model, criterion, probe_batches, device, mode="max")
            gc_history.append(gc_value)
            print(f"[Phase 1] Gradient confusion (max cosine similarity): {gc_value:.4f}")

        recorder.log_epoch(
            epoch=epoch, learning_rate=current_lr, threshold=None, phase="phase1",
            train_loss=epoch_loss, train_accuracy=epoch_accuracy,
            val_loss=val_loss, val_accuracy=accuracy,
            samples_used=samples_used, samples_total=total_samples,
            time_seconds=time_per_epoch[-1], gradient_confusion=gc_value,
        )

        if gc_value is not None and (epoch + 1) % gc_check_interval == 0 and len(gc_history) > 1:
            if is_outlier(gc_history[:-1], gc_value, threshold=gc_outlier_threshold, mode="low"):
                critical_period_end_epoch = epoch + 1
                print(f"\nCritical period ended at epoch {epoch + 1}: gradient confusion dropped as a low outlier.\n")
                break

    if critical_period_end_epoch is None:
        critical_period_end_epoch = phase1_epochs_run
    recorder.update_metadata(critical_period_end_epoch=critical_period_end_epoch)
    remaining_epochs = epochs - critical_period_end_epoch
    dbpd_epochs = max(0, remaining_epochs - final_revision_epochs)
    revision_epochs = remaining_epochs - dbpd_epochs
    print(f"Phase 1 (critical period) ran for {critical_period_end_epoch} epoch(s). "
          f"Phase 2 will run for {remaining_epochs} epoch(s): {dbpd_epochs} DBPD epoch(s) "
          f"(no augmentation) followed by {revision_epochs} full-dataset revision epoch(s).")

    # ---------------- Phase 2: DBPD, no augmentation, same model/optimizer state ----------------
    # The final `final_revision_epochs` epochs of the total budget switch to
    # training on the full, unfiltered dataset - mirroring the "revision"
    # tail of selective_gradient.TrainRevision.train_with_revision (there,
    # everything from epoch `start_revision` onward is unfiltered).
    for local_epoch in range(remaining_epochs):
        epoch = critical_period_end_epoch + local_epoch
        is_revision_epoch = epoch >= (epochs - final_revision_epochs)
        samples_used = 0
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_start_time = time.time()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        phase_label = "revision" if is_revision_epoch else "DBPD"
        progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader),
                             desc=f"Phase 2 ({phase_label}) Epoch {epoch + 1}")
        for batch_idx, (inputs, labels, sample_ids) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)  # no augmentation in phase 2

            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)

                if is_revision_epoch:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                elif threshold == 0:
                    mask = preds != labels
                else:
                    mask = correct_class < threshold

                # BatchNorm requires >=2 samples per batch in train mode; skip
                # the backward pass (and un-flag "kept") on the rare batch
                # where the threshold leaves only 0 or 1 samples.
                will_train = mask.sum().item() >= 2
                logged_mask = mask if will_train else torch.zeros_like(mask)

            recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses,
                                correct_class, logged_mask, phase="phase2_revision" if is_revision_epoch else "phase2")

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if not will_train:
                continue

            inputs_kept = inputs[mask]
            labels_kept = labels[mask]

            optimizer.zero_grad()
            outputs_kept = model(inputs_kept)
            loss = criterion(outputs_kept, labels_kept)
            loss.backward()
            optimizer.step()

            num_step += len(outputs_kept)
            samples_used += len(outputs_kept)
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        epoch_loss = running_loss / len(indexed_train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        time_per_epoch.append(time.time() - epoch_start_time)
        print(f"[Phase 2] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        accuracy, val_loss = _evaluate(model, test_loader, criterion, device)
        print(f"[Phase 2] Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step()
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)
        samples_used_per_epoch.append(samples_used)

        recorder.log_epoch(
            epoch=epoch, learning_rate=current_lr, threshold=threshold, phase="phase2",
            train_loss=epoch_loss, train_accuracy=epoch_accuracy,
            val_loss=val_loss, val_accuracy=accuracy,
            samples_used=samples_used, samples_total=total_samples,
            time_seconds=time_per_epoch[-1],
        )

    # ---------------- Final outcomes pass ----------------
    model.eval()
    final_ids, final_labels, final_preds, final_logits, final_losses, final_probs = [], [], [], [], [], []
    with torch.no_grad():
        for inputs, labels, sample_ids in tqdm(indexed_train_loader, desc="Final outcomes pass"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)
            correct_class = prob[torch.arange(labels.size(0)), labels]
            sample_losses = per_sample_loss(outputs, labels)
            final_ids.extend(sample_ids.tolist())
            final_labels.append(labels)
            final_preds.append(preds)
            final_logits.append(outputs)
            final_losses.append(sample_losses)
            final_probs.append(correct_class)
    recorder.log_outcomes(
        final_ids, torch.cat(final_labels), torch.cat(final_preds), torch.cat(final_logits),
        torch.cat(final_losses), torch.cat(final_probs),
    )

    recorder.close()
    end_time = time.time()
    log_memory(start_time, end_time)
    print(num_step)

    plot_accuracy_time_multi(
        model_name=model_name, accuracy=epoch_accuracies, time_per_epoch=time_per_epoch,
        save_path=save_path, data_file=save_path,
    )
    plot_accuracy_time_multi_test(
        model_name=model_name, accuracy=epoch_test_accuracies, time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch, threshold=threshold,
        save_path=save_path, data_file=save_path,
    )

    return model, num_step


def _final_outcomes_pass(model, indexed_train_loader, device, recorder):
    """Shared tail: one no-grad pass over the full training set logging
    final per-sample outcomes, used by every condition function below.
    """
    model.eval()
    final_ids, final_labels, final_preds, final_logits, final_losses, final_probs = [], [], [], [], [], []
    with torch.no_grad():
        for inputs, labels, sample_ids in tqdm(indexed_train_loader, desc="Final outcomes pass"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            prob = torch.softmax(outputs, dim=1)
            correct_class = prob[torch.arange(labels.size(0)), labels]
            sample_losses = per_sample_loss(outputs, labels)
            final_ids.extend(sample_ids.tolist())
            final_labels.append(labels)
            final_preds.append(preds)
            final_logits.append(outputs)
            final_losses.append(sample_losses)
            final_probs.append(correct_class)
    recorder.log_outcomes(
        final_ids, torch.cat(final_labels), torch.cat(final_preds), torch.cat(final_logits),
        torch.cat(final_losses), torch.cat(final_probs),
    )


def train_delayed_dbpd(model_name, model, train_loader, test_loader, device, epochs, save_path,
                        threshold, task, cls_num_list, onset,
                        final_revision_epochs=1, run_id=None, seed=None, dataset_name=None):
    """C5 ("delayed_dbpd") from claude_code_experiment_spec.md Sec 2 - phase
    protection: standard full-dataset training for the first `onset` epochs
    (no default - the spec requires this be supplied per-run), then vanilla
    DBPD from epoch `onset` onward - each batch filtered to samples whose
    true-class probability is below `threshold`, exactly as
    selective_gradient.TrainRevision.train_with_revision's masked branch.
    The final `final_revision_epochs` epochs of the total budget always
    train on the full, unfiltered dataset (spec Sec 1: "Final epoch:
    Full-dataset revision epoch in ALL dropout conditions").

    Unlike train_critical_periods_dbpd, `onset` is a fixed, caller-supplied
    epoch (no gradient-confusion detection), and augmentation is NOT varied
    between phases - the spec holds augmentation constant across every
    condition (Sec 1), so this is intentionally plainer than the CP variant.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    run_id = run_id if run_id else make_run_id(model_name)
    indexed_train_loader = DataLoader(IndexedDataset(train_loader.dataset), batch_size=train_loader.batch_size, shuffle=True)
    recorder = MetricsRecorder(
        model_name, save_path, run_id=run_id, seed=seed, dataset=dataset_name, architecture=model_name,
        optimizer_name="AdamW", learning_rate=3e-4, total_epochs=epochs, batch_size=train_loader.batch_size,
        threshold=threshold, start_revision=None, mode="delayed_dbpd",
        onset=onset, final_revision_epochs=final_revision_epochs,
    )

    epoch_losses, epoch_accuracies = [], []
    epoch_test_accuracies, epoch_test_losses = [], []
    time_per_epoch = []
    samples_used_per_epoch = []
    num_step = 0
    start_time = time.time()
    epoch0_sample_ids, epoch0_labels = [], []

    for epoch in range(epochs):
        is_revision_epoch = epoch >= (epochs - final_revision_epochs)
        is_delay_phase = epoch < onset
        samples_used = 0
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_start_time = time.time()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        if is_delay_phase:
            phase_label = "delay"
        elif is_revision_epoch:
            phase_label = "revision"
        else:
            phase_label = "dbpd"

        progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader),
                             desc=f"delayed_dbpd ({phase_label}) Epoch {epoch + 1}")
        for batch_idx, (inputs, labels, sample_ids) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)

                if is_delay_phase or is_revision_epoch:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                elif threshold == 0:
                    mask = preds != labels
                else:
                    mask = correct_class < threshold

                # BatchNorm requires >=2 samples per batch in train mode; skip
                # the backward pass (and un-flag "kept") on the rare batch
                # where the threshold leaves only 0 or 1 samples.
                will_train = mask.sum().item() >= 2
                logged_mask = mask if will_train else torch.zeros_like(mask)

            recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses,
                                correct_class, logged_mask, phase=phase_label)
            if epoch == 0:
                epoch0_sample_ids.extend(sample_ids.tolist())
                epoch0_labels.extend(labels.detach().cpu().tolist())

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if not will_train:
                continue

            inputs_kept = inputs[mask]
            labels_kept = labels[mask]

            optimizer.zero_grad()
            outputs_kept = model(inputs_kept)
            loss = criterion(outputs_kept, labels_kept)
            loss.backward()
            optimizer.step()

            num_step += len(outputs_kept)
            samples_used += len(outputs_kept)
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        if epoch == 0:
            recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

        epoch_loss = running_loss / len(indexed_train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        time_per_epoch.append(time.time() - epoch_start_time)
        print(f"[{phase_label}] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        accuracy, val_loss = _evaluate(model, test_loader, criterion, device)
        print(f"[{phase_label}] Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step()
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)
        samples_used_per_epoch.append(samples_used)

        recorder.log_epoch(
            epoch=epoch, learning_rate=current_lr, threshold=None if is_delay_phase else threshold,
            phase=phase_label, train_loss=epoch_loss, train_accuracy=epoch_accuracy,
            val_loss=val_loss, val_accuracy=accuracy,
            samples_used=samples_used, samples_total=total_samples,
            time_seconds=time_per_epoch[-1],
        )

    _final_outcomes_pass(model, indexed_train_loader, device, recorder)
    recorder.close()
    log_memory(start_time, time.time())
    print(num_step)

    plot_accuracy_time_multi(
        model_name=model_name, accuracy=epoch_accuracies, time_per_epoch=time_per_epoch,
        save_path=save_path, data_file=save_path,
    )
    plot_accuracy_time_multi_test(
        model_name=model_name, accuracy=epoch_test_accuracies, time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch, threshold=threshold,
        save_path=save_path, data_file=save_path,
    )

    return model, num_step


def train_soft_dbpd(model_name, model, train_loader, test_loader, device, epochs, save_path,
                     threshold, task, cls_num_list, keep_rate=0.5,
                     final_revision_epochs=1, run_id=None, seed=None, dataset_name=None):
    """ bias correction for vanilla DBPD. Each epoch (except the final
    `final_revision_epochs` full-dataset revision epochs), every batch is
    split by true-class probability into hard (< `threshold`) and easy
    (>= `threshold`) samples. Hard samples always train with weight 1; from
    the easy samples, a fresh uniform-random `keep_rate` fraction is
    retained each batch and trained with weight `1 / keep_rate` (InfoBatch-
    style inverse-probability rescaling), so the easy group's expected
    gradient contribution stays unbiased relative to training on the full
    easy set, while cutting the actual compute spent on it.

    Per the spec: "loss must be computed with reduction='none', multiplied
    by a per-sample weight vector, then meaned" - done here by meaning over
    the kept (hard + retained-easy) subset only, matching InfoBatch's own
    rescaled-mean formulation. Sampling here is per-batch rather than a
    single pass over the whole epoch's easy set upfront, since every other
    training loop in this codebase (train_with_revision, the Phase 2 above)
    is online/per-batch; since batches are drawn via shuffle=True, retaining
    a `keep_rate` fraction of each batch's easy set is statistically
    equivalent in expectation to retaining that fraction of the full
    epoch's easy set.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    # ---------------- Checkpoint/resume ----------------
    # A crash (segfault, driver reset, OOM) kills the whole process, so
    # nothing inside this function's own try/except can catch it - recovery
    # has to happen across process restarts: an external retry loop
    # relaunches this same command, and this block picks the run back up
    # from the last epoch boundary instead of starting over from epoch 0.
    ckpt_path = f"{save_path}_checkpoint.pt"
    start_epoch = 0
    epoch_losses, epoch_accuracies = [], []
    epoch_test_accuracies, epoch_test_losses = [], []
    time_per_epoch = []
    samples_used_per_epoch = []
    num_step = 0
    pending_recorder_state = None

    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        run_id = checkpoint["run_id"]
        start_epoch = checkpoint["epoch"] + 1
        epoch_losses = checkpoint["epoch_losses"]
        epoch_accuracies = checkpoint["epoch_accuracies"]
        epoch_test_accuracies = checkpoint["epoch_test_accuracies"]
        epoch_test_losses = checkpoint["epoch_test_losses"]
        time_per_epoch = checkpoint["time_per_epoch"]
        samples_used_per_epoch = checkpoint["samples_used_per_epoch"]
        num_step = checkpoint["num_step"]
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if torch.cuda.is_available() and checkpoint.get("cuda_rng_state") is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        np.random.set_state(checkpoint["numpy_rng_state"])
        random.setstate(checkpoint["python_rng_state"])
        pending_recorder_state = checkpoint["recorder_state"]
        print(f"[soft_dbpd] Resumed from checkpoint: epoch {start_epoch}/{epochs} (run_id={run_id})")
    else:
        run_id = run_id if run_id else make_run_id(model_name)

    indexed_train_loader = DataLoader(IndexedDataset(train_loader.dataset), batch_size=train_loader.batch_size, shuffle=True)
    recorder = MetricsRecorder(
        model_name, save_path, run_id=run_id, seed=seed, dataset=dataset_name, architecture=model_name,
        optimizer_name="AdamW", learning_rate=3e-4, total_epochs=epochs, batch_size=train_loader.batch_size,
        threshold=threshold, start_revision=None, mode="soft_dbpd",
        keep_rate=keep_rate, final_revision_epochs=final_revision_epochs,
    )
    if pending_recorder_state is not None:
        recorder.import_state(pending_recorder_state)

    start_time = time.time()
    epoch0_sample_ids, epoch0_labels = [], []

    for epoch in range(start_epoch, epochs):
        is_revision_epoch = epoch >= (epochs - final_revision_epochs)
        samples_used = 0
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_start_time = time.time()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        phase_label = "revision" if is_revision_epoch else "soft_dbpd"
        progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader),
                             desc=f"soft_dbpd ({phase_label}) Epoch {epoch + 1}")
        for batch_idx, (inputs, labels, sample_ids) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            model.eval()
            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)

                weights = torch.ones_like(correct_class)
                if is_revision_epoch:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                else:
                    hard_mask = (preds != labels) if threshold == 0 else (correct_class < threshold)
                    easy_mask = ~hard_mask
                    retained_easy_mask = easy_mask & (torch.rand_like(correct_class) < keep_rate)
                    mask = hard_mask | retained_easy_mask
                    weights[retained_easy_mask] = 1.0 / keep_rate

                # BatchNorm requires >=2 samples per batch in train mode; skip
                # the backward pass (and un-flag "kept") on the rare batch
                # where filtering leaves only 0 or 1 samples.
                will_train = mask.sum().item() >= 2
                logged_mask = mask if will_train else torch.zeros_like(mask)
            model.train()  # restore before the real forward/backward so BN updates running stats only from inputs_kept

            recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses,
                                correct_class, logged_mask, phase=phase_label)
            if epoch == 0:
                epoch0_sample_ids.extend(sample_ids.tolist())
                epoch0_labels.extend(labels.detach().cpu().tolist())

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if not will_train:
                continue

            original_batch_size = labels.size(0)       

            inputs_kept = inputs[mask]
            labels_kept = labels[mask]
            weights_kept = weights[mask]

            optimizer.zero_grad()
            outputs_kept = model(inputs_kept)
            per_sample = per_sample_loss(outputs_kept, labels_kept)
            loss = (per_sample * weights_kept).sum() / original_batch_size
            loss.backward()
            optimizer.step()

            num_step += len(outputs_kept)
            samples_used += len(outputs_kept)
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        if epoch == 0:
            recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

        epoch_loss = running_loss / len(indexed_train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        time_per_epoch.append(time.time() - epoch_start_time)
        print(f"[{phase_label}] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        accuracy, val_loss = _evaluate(model, test_loader, criterion, device)
        print(f"[{phase_label}] Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step()
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)
        samples_used_per_epoch.append(samples_used)

        recorder.log_epoch(
            epoch=epoch, learning_rate=current_lr, threshold=threshold, phase=phase_label,
            train_loss=epoch_loss, train_accuracy=epoch_accuracy,
            val_loss=val_loss, val_accuracy=accuracy,
            samples_used=samples_used, samples_total=total_samples,
            time_seconds=time_per_epoch[-1],
        )
        # This epoch's samples file is finalized (footer written, readable)
        # before the checkpoint is saved, so a crash on the very next line
        # never leaves an unreadable half-written samples file behind.
        recorder.finalize_epoch_samples()

        torch.save({
            "epoch": epoch,
            "run_id": run_id,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "epoch_losses": epoch_losses,
            "epoch_accuracies": epoch_accuracies,
            "epoch_test_accuracies": epoch_test_accuracies,
            "epoch_test_losses": epoch_test_losses,
            "time_per_epoch": time_per_epoch,
            "samples_used_per_epoch": samples_used_per_epoch,
            "num_step": num_step,
            "recorder_state": recorder.export_state(),
            "torch_rng_state": torch.get_rng_state(),
            "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "numpy_rng_state": np.random.get_state(),
            "python_rng_state": random.getstate(),
        }, ckpt_path)

    _final_outcomes_pass(model, indexed_train_loader, device, recorder)
    recorder.merge_epoch_samples()
    recorder.close()
    log_memory(start_time, time.time())
    print(num_step)

    plot_accuracy_time_multi(
        model_name=model_name, accuracy=epoch_accuracies, time_per_epoch=time_per_epoch,
        save_path=save_path, data_file=save_path,
    )
    plot_accuracy_time_multi_test(
        model_name=model_name, accuracy=epoch_test_accuracies, time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch, threshold=threshold,
        save_path=save_path, data_file=save_path,
    )

    # Training completed all `epochs` without crashing - no longer need the
    # resume checkpoint, and leaving it behind would make a *later*,
    # unrelated run at the same save_path silently resume from this one.
    if os.path.exists(ckpt_path):
        os.remove(ckpt_path)

    return model, num_step


def train_combined(model_name, model, train_loader, test_loader, device, epochs, save_path,
                    threshold, task, cls_num_list, onset, keep_rate=0.1,
                    final_revision_epochs=1, run_id=None, seed=None, dataset_name=None):
    """C6 ("combined") from claude_code_experiment_spec.md Sec 2: C5's
    delayed onset plus C4's soft correction. Full-dataset training for the
    first `onset` epochs (no default - required), then soft-DBPD (see
    train_soft_dbpd's docstring for the weighting mechanism) for the rest of
    the budget. The final `final_revision_epochs` epochs always train on the
    full, unfiltered dataset regardless of phase.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.98)

    run_id = run_id if run_id else make_run_id(model_name)
    indexed_train_loader = DataLoader(IndexedDataset(train_loader.dataset), batch_size=train_loader.batch_size, shuffle=True)
    recorder = MetricsRecorder(
        model_name, save_path, run_id=run_id, seed=seed, dataset=dataset_name, architecture=model_name,
        optimizer_name="AdamW", learning_rate=3e-4, total_epochs=epochs, batch_size=train_loader.batch_size,
        threshold=threshold, start_revision=None, mode="combined",
        onset=onset, keep_rate=keep_rate, final_revision_epochs=final_revision_epochs,
    )

    epoch_losses, epoch_accuracies = [], []
    epoch_test_accuracies, epoch_test_losses = [], []
    time_per_epoch = []
    samples_used_per_epoch = []
    num_step = 0
    start_time = time.time()
    epoch0_sample_ids, epoch0_labels = [], []

    for epoch in range(epochs):
        is_revision_epoch = epoch >= (epochs - final_revision_epochs)
        is_delay_phase = epoch < onset
        samples_used = 0
        current_lr = optimizer.param_groups[0]["lr"]
        model.train()
        epoch_start_time = time.time()
        running_loss, total_correct, total_samples = 0.0, 0, 0

        if is_delay_phase:
            phase_label = "delay"
        elif is_revision_epoch:
            phase_label = "revision"
        else:
            phase_label = "soft_dbpd"

        progress_bar = tqdm(enumerate(indexed_train_loader), total=len(indexed_train_loader),
                             desc=f"combined ({phase_label}) Epoch {epoch + 1}")
        for batch_idx, (inputs, labels, sample_ids) in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)

            with torch.no_grad():
                outputs = model(inputs)
                preds = torch.argmax(outputs, dim=1)
                prob = torch.softmax(outputs, dim=1)
                correct_class = prob[torch.arange(labels.size(0)), labels]
                sample_losses = per_sample_loss(outputs, labels)

                weights = torch.ones_like(correct_class)
                if is_delay_phase or is_revision_epoch:
                    mask = torch.ones_like(labels, dtype=torch.bool)
                else:
                    hard_mask = (preds != labels) if threshold == 0 else (correct_class < threshold)
                    easy_mask = ~hard_mask
                    retained_easy_mask = easy_mask & (torch.rand_like(correct_class) < keep_rate)
                    mask = hard_mask | retained_easy_mask
                    weights[retained_easy_mask] = 1.0 / keep_rate

                will_train = mask.sum().item() >= 2
                logged_mask = mask if will_train else torch.zeros_like(mask)

            recorder.log_batch(epoch, sample_ids.tolist(), labels, preds, outputs, sample_losses,
                                correct_class, logged_mask, phase=phase_label)
            if epoch == 0:
                epoch0_sample_ids.extend(sample_ids.tolist())
                epoch0_labels.extend(labels.detach().cpu().tolist())

            total_correct += (preds == labels).sum().item()
            total_samples += labels.size(0)

            if not will_train:
                continue

            inputs_kept = inputs[mask]
            labels_kept = labels[mask]
            weights_kept = weights[mask]

            optimizer.zero_grad()
            outputs_kept = model(inputs_kept)
            per_sample = per_sample_loss(outputs_kept, labels_kept)
            loss = (per_sample * weights_kept).mean()
            loss.backward()
            optimizer.step()

            num_step += len(outputs_kept)
            samples_used += len(outputs_kept)
            running_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        if epoch == 0:
            recorder.log_identifiers(epoch0_sample_ids, epoch0_labels)

        epoch_loss = running_loss / len(indexed_train_loader)
        epoch_accuracy = total_correct / total_samples if total_samples > 0 else 0
        epoch_losses.append(epoch_loss)
        epoch_accuracies.append(epoch_accuracy)
        time_per_epoch.append(time.time() - epoch_start_time)
        print(f"[{phase_label}] Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.4f}")

        accuracy, val_loss = _evaluate(model, test_loader, criterion, device)
        print(f"[{phase_label}] Epoch {epoch + 1}/{epochs}, Test Accuracy: {accuracy:.4f}, Test Loss: {val_loss:.4f}")
        scheduler.step()
        epoch_test_accuracies.append(accuracy)
        epoch_test_losses.append(val_loss)
        samples_used_per_epoch.append(samples_used)

        recorder.log_epoch(
            epoch=epoch, learning_rate=current_lr, threshold=None if is_delay_phase else threshold,
            phase=phase_label, train_loss=epoch_loss, train_accuracy=epoch_accuracy,
            val_loss=val_loss, val_accuracy=accuracy,
            samples_used=samples_used, samples_total=total_samples,
            time_seconds=time_per_epoch[-1],
        )

    _final_outcomes_pass(model, indexed_train_loader, device, recorder)
    recorder.close()
    log_memory(start_time, time.time())
    print(num_step)

    plot_accuracy_time_multi(
        model_name=model_name, accuracy=epoch_accuracies, time_per_epoch=time_per_epoch,
        save_path=save_path, data_file=save_path,
    )
    plot_accuracy_time_multi_test(
        model_name=model_name, accuracy=epoch_test_accuracies, time_per_epoch=time_per_epoch,
        samples_per_epoch=samples_used_per_epoch, threshold=threshold,
        save_path=save_path, data_file=save_path,
    )

    return model, num_step
