import os
import json
import shutil
import time

import torch
from torch.utils.data import DataLoader

from src.training.lightningmodule.base_lightningmodule import BaseLightningModule
from src.training.lightningmodule.helper import load_ckpt
from src.datamodules.dataset.s5.dataset_s5_waveform import DatasetS5Waveform
from src.evaluation.evaluation import evaluate_separator_with_tagger
from src.utils import LABELS, initialize_config
from src.models.s5.s5 import S5
from src.paths import LOCAL_DATA_PATH

class BaseLabelQueriedSeparationModule(BaseLightningModule):
    def __init__(
            self,
            model: dict,
            loss: dict,
            optimizer: dict,
            lr_scheduler: dict = None,
            is_validation: bool = False,
            metric: dict = None,
            soundbeam: dict = None,
            evaluation_tagger_model: dict = None,
    ):
        if soundbeam is not None and soundbeam['apply']:
            model['args']['soundbeam'] = soundbeam  # pass the soundbeam config to the model

        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_validation=is_validation,
            metric=metric,
        )

        self.embedding_model = None
        self.label_set = "dcase2025t4"
        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False, dtype=torch.float32)

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
        batch_size = 1
        self.test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=test_dataset.collate_fn,
            num_workers=batch_size * 2
        )

        self.evaluation_tagger_model = evaluation_tagger_model
        self.eval_checkpoint = self.evaluation_tagger_model['args']['checkpoint']

        if soundbeam is None:
            soundbeam = {'apply': False}

        self.soundbeam = soundbeam
        self.embedding_model_is_trainable = False

        if self.soundbeam['apply'] or model["args"]["use_time_film"]:
            print("Soundbeam is enabled")
            self.setup_soundbeam(self.soundbeam['tagger_model'], self.soundbeam['trainable'], self.soundbeam["lr"], self.soundbeam["lr_decay_factor"])


    def setup_soundbeam(self, tagger_model_config, trainable=False, lr=None, lr_decay_factor=1.0):
        tagger = initialize_config(tagger_model_config)

        if tagger_model_config['args']['checkpoint'] == 'baseline':
            tagger_ckpt = "checkpoint/m2dat.ckpt"
            load_ckpt(tagger_ckpt, tagger)

        self.embedding_model = tagger
        self.embedding_model_is_trainable = trainable

        if lr is not None and not trainable:
            raise ValueError("If the embedding model is not trainable, no learning rate should be specified.")

        if trainable:
            print("Embedding model is trainable")
            self.embedding_model.train()

            if lr is None:
                raise ValueError("Learning rate must be specified if the embedding model is trainable.")

            params = self.optimizer_config['args']['params']

            # add the embedding model parameters to the optimizer
            if hasattr(self.embedding_model.model, "separate_params"):
                embedding_model_params = self.embedding_model.model.separate_params()
            else:
                raise NotImplementedError("The base model has no 'separate_params' method!'")

            # apply lr_decay of lr_decay_factor to the embedding model parameters
            scaled_lrs = [lr * (lr_decay_factor ** i) for i in range(1, len(embedding_model_params) + 1)]

            scaled_embedding_model_params = [{"params": embedding_model_params[i], "lr": scaled_lrs[i]} for i in
                                       range(len(embedding_model_params))]

            self.optimizer_config['args']['params'] = params + scaled_embedding_model_params
            self.optimizer = initialize_config(self.optimizer_config)

        else:
            self.embedding_model.eval()

            for param in self.embedding_model.parameters():
                param.requires_grad = False


    def _get_tagger_predictions(self, s5_model, dataloader):
        """Generate and collect tagger predictions for all samples in the dataloader."""
        predictions = []
        phase = 'val'
        s5_model.eval()

        for batch in dataloader:
            mixtures = batch['mixture'].to('cpu')  # [bs, nch, wlen]

            with torch.no_grad():
                output = s5_model.predict_label(mixtures)
                labels = output['label']  # List of label lists
            if 'soundscape_name' in batch:
                phase = 'test'
                soundscape_names = batch['soundscape_name']
                for name, label_list in zip(soundscape_names, labels):
                    predictions.append((name, label_list))
            else:
                predictions.extend(labels)

        if phase == 'test':
            return dict(predictions)

        return predictions


    def setup(self, stage=None):
        """Save tagger predictions for validation and test sets at the start of training."""

        if self.trainer.is_global_zero:
            if stage == "fit" or stage is None:

                # Define file names
                test_file = f'test_{self.eval_checkpoint}_tagger_predictions.json'
                val_file = f'val_{self.eval_checkpoint}_tagger_predictions.json'

                # Handle test file
                test_local_path = os.path.join(LOCAL_DATA_PATH, test_file)
                test_resource_path = os.path.join('resources', test_file)
                if os.path.exists(test_resource_path) and not os.path.exists(test_local_path):
                    shutil.copy(test_resource_path, LOCAL_DATA_PATH)

                # Handle validation file
                val_local_path = os.path.join(LOCAL_DATA_PATH, val_file)
                val_resource_path = os.path.join('resources', val_file)
                if os.path.exists(val_resource_path) and not os.path.exists(val_local_path):
                    shutil.copy(val_resource_path, LOCAL_DATA_PATH)

                create_test_file = not os.path.exists(test_resource_path) and not os.path.exists(test_local_path)
                create_val_file = not os.path.exists(val_resource_path) and not os.path.exists(val_local_path)

                if create_val_file or create_test_file:
                    # delete existing files if they exist
                    if os.path.exists(test_local_path):
                        os.remove(test_local_path)

                    if os.path.exists(val_local_path):
                        os.remove(val_local_path)

                    print("-" * 20, "\nCreating tagger predictions for test set...\n", "-" * 20, flush=True)

                    # Define tagger configuration
                    # for evaluating, the baseline M2D model is standard! This is hardcoded (and has nothing to do with the model used for Soundbeam!)

                    if self.eval_checkpoint == 'baseline':
                        tagger_ckpt = "checkpoint/m2dat.ckpt"
                    else:
                        raise ValueError("Tagger checkpoint not found or not specified.")

                    # Create S5 model with only the tagger (no separator needed)
                    s5_model = S5(
                        label_set=self.label_set,
                        tagger_config=self.evaluation_tagger_model,
                        separator_config=None,
                        tagger_ckpt=tagger_ckpt,
                        separator_ckpt=None
                    )

                    # Save validation predictions
                    val_dataloader = self.trainer.datamodule.val_dataloader()

                    if val_dataloader:
                        val_predictions = self._get_tagger_predictions(s5_model, val_dataloader)
                        with open(os.path.join(LOCAL_DATA_PATH, f'val_{self.eval_checkpoint}_tagger_predictions.json'), 'w') as f:
                            json.dump(val_predictions, f, indent=4)

                    # Save test predictions
                    test_predictions = self._get_tagger_predictions(s5_model, self.test_dataloader)
                    with open(os.path.join(LOCAL_DATA_PATH, f'test_{self.eval_checkpoint}_tagger_predictions.json'), 'w') as f:
                        json.dump(test_predictions, f, indent=4)

                    del s5_model

    def on_validation_epoch_end(self):
        is_last_epoch = (self.current_epoch + 1) == self.trainer.max_epochs
        if self.trainer.is_global_zero and (self.current_epoch % 3 == 0 or is_last_epoch):
            # Load saved predictions
            with open(os.path.join(LOCAL_DATA_PATH, f'val_{self.eval_checkpoint}_tagger_predictions.json'), 'r') as f:
                val_predictions = json.load(f)

            with open(os.path.join(LOCAL_DATA_PATH, f'test_{self.eval_checkpoint}_tagger_predictions.json'), 'r') as f:
                test_predictions = json.load(f)

            val_dataloader = self.trainer.datamodule.val_dataloader()

            # measure the time
            start_time = time.time()

            # Evaluate using saved predictions
            val_result = evaluate_separator_with_tagger(
                separator=self.model,
                embedding_model=self.embedding_model if self.embedding_model else None,
                dataloader=val_dataloader,
                predictions=val_predictions,
                label_onehots=self.label_onehots,
                label_len=self.label_len,
                phase='val'
            )
            end_time = time.time()
            print(f"Evaluation time validation set: {end_time - start_time:.2f} seconds")

            start_time = time.time()
            test_result = evaluate_separator_with_tagger(
                separator=self.model,
                embedding_model=self.embedding_model if self.embedding_model else None,
                dataloader=self.test_dataloader,
                predictions=test_predictions,
                label_onehots=self.label_onehots,
                label_len=self.label_len,
                phase='test'
            )
            end_time = time.time()
            print(f"Evaluation time test set: {end_time - start_time:.2f} seconds")

            ca_sdr_val = val_result['ca_sdr'].item() if torch.is_tensor(val_result['ca_sdr']) else val_result['ca_sdr']
            ca_sdr_test = test_result['ca_sdr'].item() if torch.is_tensor(test_result['ca_sdr']) else test_result[
                'ca_sdr']
            del val_result, test_result
            torch.cuda.empty_cache()

            # Log metrics
            self.log('epoch_val/ca_sdr', ca_sdr_val, on_epoch=True, sync_dist=False)  # sync_dist=False is okay here
            self.log('epoch_test/ca_sdr', ca_sdr_test, on_epoch=True, sync_dist=False)

    def on_test_epoch_end(self):
        with open(os.path.join(LOCAL_DATA_PATH, f'test_{self.eval_checkpoint}_tagger_predictions.json'), 'r') as f:
            test_predictions = json.load(f)

        start_time = time.time()
        test_result = evaluate_separator_with_tagger(
            separator=self.model,
            embedding_model=self.embedding_model if self.embedding_model else None,
            dataloader=self.test_dataloader,
            predictions=test_predictions,
            label_onehots=self.label_onehots,
            label_len=self.label_len,
            phase='test'
        )
        end_time = time.time()
        print(f"Evaluation time test set: {end_time - start_time:.2f} seconds")

        ca_sdr_test = test_result['ca_sdr'].item() if torch.is_tensor(test_result['ca_sdr']) else test_result[
            'ca_sdr']
        del test_result
        torch.cuda.empty_cache()

        self.log('test/ca_sdr', ca_sdr_test, on_epoch=True, sync_dist=False)  # sync_dist=False is okay here

        print("Test set evaluation completed. CA-SDR:", ca_sdr_test)
