import time
import os
from torch.utils.data import DataLoader
import torch
import numpy as np

from .base_lightningmodule import BaseLightningModule
from src.models.s5.s5 import S5
from src.evaluation.evaluation import evaluate_tagger, evaluate_sed
from src.datamodules.dataset.s5.dataset_s5_waveform import DatasetS5Waveform
from src.paths import LOCAL_DATA_PATH
from src.utils import LABELS


def mixup(data, targets, strong_targets=None, alpha=0.2, beta=0.2):
    with torch.no_grad():
        batch_size = data.size(0)
        c = np.random.beta(alpha, beta, size=batch_size)
        c = np.maximum(c, 1 - c)

        perm = torch.randperm(batch_size)

        # Mix audio
        c_tensor = torch.tensor(c, dtype=data.dtype, device=data.device).view(batch_size, *([1] * (data.ndim - 1)))
        mixed_data = c_tensor * data + (1 - c_tensor) * data[perm]

        # Mix weak (clip-level) labels
        ct_tensor = c_tensor.view(batch_size, *([1] * (targets.ndim - 1)))
        mixed_targets = ct_tensor * targets + (1 - ct_tensor) * targets[perm]

        # Mix strong (frame-level) labels if provided
        if strong_targets is not None:
            cs_tensor = c_tensor.view(batch_size, *([1] * (strong_targets.ndim - 1)))
            mixed_strong_targets = cs_tensor * strong_targets + (1 - cs_tensor) * strong_targets[perm]
            return mixed_data, mixed_targets, mixed_strong_targets

    return mixed_data, mixed_targets


class SoundEventDetection(BaseLightningModule):
    def __init__(self, *args, **kwargs):
        mixup_p = kwargs.pop('mixup_p', 0.0)
        weak_loss_weight = kwargs.pop('weak_loss_weight', 0.5)
        super().__init__(*args, **kwargs)

        soundscape_dir = 'data/dev_set/test/soundscape'
        oracle_target_dir = 'data/dev_set/test/oracle_target'

        soundscape_dir = os.path.join(LOCAL_DATA_PATH, soundscape_dir[len('data/dev_set/'):])
        oracle_target_dir = os.path.join(LOCAL_DATA_PATH, oracle_target_dir[len('data/dev_set/'):])

        test_dataset = DatasetS5Waveform(
            soundscape_dir=soundscape_dir,
            oracle_target_dir=oracle_target_dir,
            estimate_target_dir=None,
            n_sources=3,
            label_set="dcase2025t4",
            label_vector_mode="concat",
            sr=32000
        )

        # load model and dataset
        batch_size = 2
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            num_workers=batch_size * 2
        )

        self.mixup_p = mixup_p
        self.weak_loss_weight = weak_loss_weight

    def _get_strong_labels(self, label_data, sr, duration, num_frames):
        B = len(label_data)
        C = len(next(iter(label_data[0].keys())))  # infer num_classes from one-hot key
        label_tensor = torch.zeros(B, C, num_frames)

        for sample_idx, events in enumerate(label_data):
            total_samples = sr[sample_idx] * duration[sample_idx]
            for onehot, (start, end) in events.items():
                cls_idx = torch.tensor(onehot).nonzero(as_tuple=False).item()
                start_bin = round(start / total_samples * num_frames)
                end_bin = round(end / total_samples * num_frames)

                # Clip to valid frame range
                start_bin = max(0, min(start_bin, num_frames - 1))
                end_bin = max(start_bin + 1, min(end_bin, num_frames))

                label_tensor[sample_idx, cls_idx, start_bin:end_bin] = 1.0

        return label_tensor

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'waveform': batch_data_dict['mixture'],
            # [bs, nch, wlen] = [bs, 4, 320_000]  # comment from baseline corrected!
        }

        label_strong = self._get_strong_labels(batch_data_dict['on_offset'],
                                               batch_data_dict['sr'],
                                               batch_data_dict['duration'],
                                               self.model.seq_len)
        label_strong = label_strong.to(batch_data_dict['label_vector'].device)

        if self.mixup_p > 0 and torch.rand(1) < self.mixup_p:
            input_dict['waveform'], batch_data_dict['label_vector'], label_strong = mixup(input_dict['waveform'],
                                                                                          batch_data_dict[
                                                                                              'label_vector'],
                                                                                          strong_targets=label_strong)

        output_dict = self.model(input_dict)
        output_strong = {"probabilities": output_dict['probabilities_strong']}
        output_weak = {"probabilities": output_dict['probabilities']}

        target_strong = {'probabilities': label_strong}
        target_weak = {'probabilities': batch_data_dict['label_vector']}

        loss_strong = self.loss_func(output_strong, target_strong)["loss"]
        loss_weak = self.loss_func(output_weak, target_weak)["loss"]

        loss = self.weak_loss_weight * loss_weak + (1 - self.weak_loss_weight) * loss_strong
        loss_dict = {"loss": loss, "loss_weak": loss_weak, "loss_strong": loss_strong}
        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'waveform': batch_data_dict['mixture'],
            # [bs, nch, wlen] = [bs, 4, 320_000]  # comment from baseline corrected!
        }

        label_strong = self._get_strong_labels(batch_data_dict['on_offset'],
                                               batch_data_dict['sr'],
                                               batch_data_dict['duration'],
                                               self.model.seq_len)
        label_strong = label_strong.to(batch_data_dict['label_vector'].device)

        output_dict = self.model(input_dict)
        output_strong = {"probabilities": output_dict['probabilities_strong']}
        output_weak = {"probabilities": output_dict['probabilities']}

        target_strong = {'probabilities': label_strong}
        target_weak = {'probabilities': batch_data_dict['label_vector']}

        loss_strong = self.loss_func(output_strong, target_strong)["loss"]
        loss_weak = self.loss_func(output_weak, target_weak)["loss"]

        loss = self.weak_loss_weight * loss_weak + (1 - self.weak_loss_weight) * loss_strong
        loss_dict = {"loss": loss, "loss_weak": loss_weak, "loss_strong": loss_strong}

        target_dict = {'probabilities': batch_data_dict['label_vector']}

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if self.metric_func:  # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():  # metric return [bs] for better calculation of mean
                loss_dict[k] = v.mean().item()  # torch tensor size [bs]

        return batchsize, loss_dict

    def on_validation_epoch_end(self):
        if self.trainer.is_global_zero and self.current_epoch % 1 == 0:
            # data loaders
            val_dataloader = self.trainer.datamodule.val_dataloader()
            test_dataloader = self.test_dataloader

            s5_model = S5(
                label_set='dcase2025t4',  # hardcoded for this evaluation
                tagger_config=None,
                separator_config=None,
                tagger_ckpt=None,
                separator_ckpt=None,
                tagger_obj=self.model
            )

            # measure the time
            start_time = time.time()

            # Evaluate using saved predictions
            val_result = evaluate_sed(
                sed_model=s5_model,
                dataloader=val_dataloader,
                class_labels=LABELS['dcase2025t4'],
                phase="val"
            )
            end_time = time.time()
            print(f"Evaluation time validation set: {end_time - start_time:.2f} seconds")

            start_time = time.time()
            test_result = evaluate_tagger(
                tagger=s5_model,
                dataloader=test_dataloader,
                phase="test"
            )
            end_time = time.time()
            print(f"Evaluation time test set: {end_time - start_time:.2f} seconds")

            # --- Logging ---
            # Accuracy per threshold
            for pthre, acc in val_result["accuracy_per_threshold"].items():
                self.log(f"epoch_val/val_acc@{pthre:.1f}", acc, on_epoch=True, sync_dist=False)
            for p in test_result:
                self.log(f"epoch_test/test_acc@{p:.1f}", test_result[p], on_epoch=True, sync_dist=False)

    def test_step(self, batch_data_dict, batch_idx):
        """Test step for sound event detection model."""
        self.model.eval()
        
        batchsize, loss_dict = self.test_step_processing(batch_data_dict, batch_idx)
        
        # Log all items in loss_dict
        step_dict = {f'step_test/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoc_dict = {f'epoch_test/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

    def test_step_processing(self, batch_data_dict, batch_idx):
        """Test step processing for sound event detection model."""
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'waveform': batch_data_dict['mixture'],
            # [bs, nch, wlen] = [bs, 4, 320_000]
        }

        label_strong = self._get_strong_labels(batch_data_dict['on_offset'],
                                               batch_data_dict['sr'],
                                               batch_data_dict['duration'],
                                               self.model.seq_len)
        label_strong = label_strong.to(batch_data_dict['label_vector'].device)

        output_dict = self.model(input_dict)
        output_strong = {"probabilities": output_dict['probabilities_strong']}
        output_weak = {"probabilities": output_dict['probabilities']}

        target_strong = {'probabilities': label_strong}
        target_weak = {'probabilities': batch_data_dict['label_vector']}

        loss_strong = self.loss_func(output_strong, target_strong)["loss"]
        loss_weak = self.loss_func(output_weak, target_weak)["loss"]

        loss = self.weak_loss_weight * loss_weak + (1 - self.weak_loss_weight) * loss_strong
        loss_dict = {"loss": loss, "loss_weak": loss_weak, "loss_strong": loss_strong}

        target_dict = {'probabilities': batch_data_dict['label_vector']}

        loss_dict = {k: v.item() for k, v in loss_dict.items()}
        if self.metric_func:  # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k, v in metric.items():  # metric return [bs] for better calculation of mean
                loss_dict[k] = v.mean().item()  # torch tensor size [bs]

        return batchsize, loss_dict

    def on_test_epoch_end(self):
        """Test epoch end processing for comprehensive evaluation."""
        if self.trainer.is_global_zero:
            # Get test dataloader
            test_dataloader = self.test_dataloader

            s5_model = S5(
                label_set='dcase2025t4',  # hardcoded for this evaluation
                tagger_config=None,
                separator_config=None,
                tagger_ckpt=None,
                separator_ckpt=None,
                tagger_obj=self.model
            )

            # Measure evaluation time
            start_time = time.time()

            # Evaluate using comprehensive SED evaluation
            test_result = evaluate_sed(
                sed_model=s5_model,
                dataloader=test_dataloader,
                class_labels=LABELS['dcase2025t4'],
                phase="test"
            )
            end_time = time.time()
            print(f"Evaluation time test set: {end_time - start_time:.2f} seconds")

            # Logging
            # Accuracy per threshold
            for pthre, acc in test_result["accuracy_per_threshold"].items():
                self.log(f"test_acc@{pthre:.1f}", acc, on_epoch=True, sync_dist=False)
            
            # Segment-based pAUC (mean)
            self.log("test_pauroc", test_result["segment_pauc_mean"], on_epoch=True, sync_dist=False)
            
            # Print summary
            print("=== Test Results Summary ===")
            for pthre, acc in test_result["accuracy_per_threshold"].items():
                print(f"Test Accuracy @ {pthre:.1f}: {acc:.2f}%")
            print(f"Test pAUC: {test_result['segment_pauc_mean']:.4f}")
            print("===========================")
