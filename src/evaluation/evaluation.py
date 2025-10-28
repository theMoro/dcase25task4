"""Evaluation utilities for DCASE 2025 Task 4 models."""
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import sed_scores_eval

from src.models.s5.s5 import S5
from src.utils import ca_metric, LABELS
from torchmetrics.functional import signal_noise_ratio as snr


def evaluate_tagger(tagger, dataloader, phase="val"):
    """Evaluate the tagger across multiple probability thresholds."""
    device = next(tagger.parameters()).device
    pthre_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {pthre: [] for pthre in pthre_values}

    for batch in tqdm(dataloader, desc=f"[{phase}] Evaluating tagger"):
        with torch.no_grad():
            output = tagger.predict_label(batch['mixture'].to(device), pthre=pthre_values, nevent_range=[1, 3])
        pred_labels_dict = output['label']
        ref_labels = batch['label']

        for pthre in pthre_values:
            pred_labels = pred_labels_dict[pthre]
            for pred, ref in zip(pred_labels, ref_labels):
                label_check = (set(pred) - {'silence'}) == (set(ref) - {'silence'})
                results[pthre].append(label_check)

    acc_per_pthre = {
        pthre: 100 * np.mean(results[pthre]) for pthre in pthre_values
    }

    for p, acc in acc_per_pthre.items():
        print(f"[{phase}] pthre={p:.1f} - Accuracy: {acc:.2f}%")

    return acc_per_pthre


def evaluate_separator_with_tagger(separator, embedding_model, dataloader, predictions, label_onehots, label_len, phase=None):
    """Evaluate the separator using precomputed tagger predictions."""
    device = next(separator.parameters()).device
    metrics = []
    prediction_idx = 0

    for batch in tqdm(dataloader, desc=f"Evaluating separator ({phase})"):
        batch_mixture = batch['mixture']  # [bs, nch, wlen]
        batch_ref_waveforms = batch['dry_sources'][:, :, 0, :]  # [bs, nsources, wlen]
        batch_ref_labels = batch['label']
        batch_size = batch_mixture.shape[0]

        if phase == 'test':
            soundscape_names = batch['soundscape_name']
            batch_est_labels = [predictions[name] for name in soundscape_names]
        else:
            batch_est_labels = predictions[prediction_idx:prediction_idx + batch_size]
            prediction_idx += batch_size

        label_vector = torch.stack([
            torch.stack([label_onehots[label] for label in labels]).flatten()
            for labels in batch_est_labels
        ])

        input_dict = {'mixture': batch_mixture.to(device), 'label_vector': label_vector.to(device)}

        if embedding_model:
            with torch.no_grad():
                out_dict = embedding_model({'waveform': input_dict['mixture']}, return_features=True)
                input_dict['embeddings'] = out_dict["features"]
                if "logits_strong" in out_dict.keys():
                    input_dict['labels_strong'] = out_dict["logits_strong"]

        with torch.no_grad():
            assert label_len == 18  # ResUNet
            output_dict = separator.forward_inference(input_dict, label_len=label_len)

        batch_est_waveforms = output_dict['waveform'].cpu()[:, :, 0, :]

        for est_lb, est_wf, ref_lb, ref_wf, mixture in zip(
            batch_est_labels,
            batch_est_waveforms,
            batch_ref_labels,
            batch_ref_waveforms,
            batch_mixture[:, 0, :]
        ):
            metric = ca_metric(est_lb, est_wf, ref_lb, ref_wf, mixture, snr)
            metrics.append(metric)

    ca_sdr_avg = np.mean(metrics)
    return {'ca_sdr': ca_sdr_avg}


def _probs_to_df(probs_2d: np.ndarray, segment_length: float, class_labels: list[str]) -> pd.DataFrame:
    """Convert probability array to DataFrame format."""
    T = probs_2d.shape[1]
    offset = np.arange(1, T+1, dtype=np.float64) * segment_length
    onset = np.concatenate([[0.0], offset[:-1]])
    data = np.vstack([onset, offset, probs_2d]).T
    df = pd.DataFrame(data, columns=['onset', 'offset'] + class_labels)
    return df


def _check_alignment(df: pd.DataFrame, fname: str, atol: float = 1e-6):
    """Check frame alignment in DataFrame."""
    offs = df['offset'].values
    ons = df['onset'].values
    if not np.allclose(offs[:-1], ons[1:], atol=atol):
        bad = np.where(~np.isclose(offs[:-1], ons[1:], atol=atol))[0][0]
        raise ValueError(
            f"Frame-alignment error in {fname} at frame {bad}: "
            f"offset[{bad}]={offs[bad]:.8f} ≠ onset[{bad+1}]={ons[bad+1]:.8f}"
        )


def evaluate_sed(sed_model, dataloader, class_labels: list[str], phase="val", segment_length=1.0, max_fpr=0.1):
    """Evaluate SED model with segment-based metrics."""
    device = next(sed_model.parameters()).device
    pthre_values = [0.1, 0.3, 0.5, 0.7, 0.9]
    results = {pthre: [] for pthre in pthre_values}

    preds_dict = {}  # fname -> DataFrame(time × class)
    ground_truth_dict = {}  # fname -> [(on, off, class), …]
    durations_dict = {}  # fname -> duration in seconds

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"[{phase}] Evaluating SED model")):
        output = sed_model.predict_label(batch['mixture'].to(device), pthre=pthre_values, nevent_range=[1, 3])
        pred_labels_dict = output['label']
        ref_labels = batch['label']

        # SED evaluation part
        probs = output["probabilities_strong"].cpu().float().numpy()  # (B, C, T)
        B, C, T = probs.shape
        assert C == len(class_labels), "class_labels length must equal C"

        for i in range(B):
            fname = f"{phase}_clip_{i + batch_idx * B}"
            segment_len_i = batch["duration"][i] / probs.shape[2]
            preds_dict[fname] = _probs_to_df(probs[i], segment_len_i, class_labels)

            gt_events = []
            for onehot_vec, (on_samp, off_samp) in batch["on_offset"][i].items():
                cls_idx = onehot_vec.nonzero(as_tuple=True)[0].item()
                gt_events.append(
                    (on_samp / batch["sr"][i], off_samp / batch["sr"][i], class_labels[cls_idx])
                )
            ground_truth_dict[fname] = gt_events
            durations_dict[fname] = batch["duration"][i]

        for pthre in pthre_values:
            pred_labels = pred_labels_dict[pthre]
            for pred, ref in zip(pred_labels, ref_labels):
                label_check = (set(pred) - {'silence'}) == (set(ref) - {'silence'})
                results[pthre].append(label_check)

    acc_per_pthre = {
        pthre: 100 * np.mean(results[pthre]) for pthre in pthre_values
    }

    for p, acc in acc_per_pthre.items():
        print(f"[{phase}] pthre={p:.1f} - Accuracy: {acc:.2f}%")

    # Check alignment
    for fname, df in preds_dict.items():
        _check_alignment(df, fname, atol=0)

    # Calculate segment-based pAUC
    paur = sed_scores_eval.segment_based.auroc(
        preds_dict,
        ground_truth_dict,
        durations_dict,
        segment_length=segment_length,
        max_fpr=max_fpr,
        num_jobs=1,
    )

    mean_pauc = paur[0]["mean"]  # macro-average over classes
    print(f"[{phase}] Segment-based pAUC @ {max_fpr:.2f} FPR : {mean_pauc:.4f}")

    return {
        "accuracy_per_threshold": acc_per_pthre,
        "segment_pauc_mean": mean_pauc,
        "segment_pauc_full": paur
    }


def evaluate_s5_model(s5_model, dataloader, phase="val"):
    """Evaluate complete S5 model (tagger + separator)."""
    device = next(s5_model.parameters()).device
    metrics = []

    for batch in tqdm(dataloader, desc=f"Evaluating S5 model ({phase})"):
        batch_mixture = batch['mixture']  # [bs, nch, wlen]
        batch_ref_waveforms = batch['dry_sources'][:, :, 0, :]  # [bs, nsources, wlen]
        batch_ref_labels = batch['label']

        with torch.no_grad():
            output = s5_model.predict_label_separate(batch_mixture.to(device), pthre=0.5, nevent_range=[1, 3])
            batch_est_labels = output['label']
            batch_est_waveforms = output['waveform'].cpu()[:, :, 0, :]

        for est_lb, est_wf, ref_lb, ref_wf, mixture in zip(
            batch_est_labels,
            batch_est_waveforms,
            batch_ref_labels,
            batch_ref_waveforms,
            batch_mixture[:, 0, :]
        ):
            metric = ca_metric(est_lb, est_wf, ref_lb, ref_wf, mixture, snr)
            metrics.append(metric)

    ca_sdr_avg = np.mean(metrics)
    print(f"[{phase}] CA-SDR: {ca_sdr_avg:.3f}")
    return {'ca_sdr': ca_sdr_avg}