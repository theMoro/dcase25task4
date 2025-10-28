from typing import Any, Callable, Dict
import lightning.pytorch as pl
import torch
from huggingface_hub import PyTorchModelHubMixin
import transformers

from src.utils import initialize_config, LABELS


def get_lr_scheduler(
        optimizer,
        num_training_steps,
        schedule_mode="exp",
        gamma: float = 0.999996,
        num_warmup_steps=20000,
        lr_end=2e-7,
):
    if schedule_mode in {"exp"}:
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    if schedule_mode in {"cosine", "cos"}:
        return transformers.get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
    if schedule_mode in {"linear"}:
        print("Linear schedule!")
        return transformers.get_polynomial_decay_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
            power=1.0,
            lr_end=lr_end,
        )
    raise RuntimeError(f"schedule_mode={schedule_mode} Unknown.")


class BaseLightningModule(pl.LightningModule, PyTorchModelHubMixin):
    def __init__(
        self,
        model: Dict,
        loss: Dict,
        optimizer: Dict,
        lr_scheduler:Dict=None,
        is_validation=False,
        metric:Dict=None,
    ):

        super().__init__()
        self.model_config = model
        self.model = initialize_config(self.model_config)

        self.loss_config = loss
        self.loss_func = initialize_config(self.loss_config)
        self.labels = {cls: i for i, cls in enumerate(LABELS["dcase2025t4"])}
        # dcase2025t4 weil es immer dieses labelset ist, oder soll das besser flexibel sein?

        self.optimizer_config = optimizer
        if self.optimizer_config.get('split_params', False):
            if hasattr(self.model, 'separate_params'):
                lr_dprnn_in_config = 'lr_dprnn' in self.optimizer_config
                if lr_dprnn_in_config:
                    lr_dprnn = self.optimizer_config['lr_dprnn'] \
                        if self.optimizer_config['lr_dprnn'] is not None \
                        else self.optimizer_config['args']['lr']

                    params = self.model.separate_params(
                        self.optimizer_config['args']['lr'],
                        lr_dprnn,
                        self.optimizer_config.get('lr_decay_factor', 1.0),
                        self.optimizer_config['args'].get('weight_decay', 0.0),
                    )
                else:
                    params = self.model.separate_params(
                        self.optimizer_config['args']['lr'],
                        self.optimizer_config.get('lr_decay_factor', 1.0),
                        self.optimizer_config['args'].get('weight_decay', 0.0),
                    )
            else:
                raise NotImplementedError(f"Model {type(self.model)} does not implement `separate_params(...)`.")
        else:
            params = self.model.parameters()

        self.optimizer_config['args']['params'] = params  # modify if some parts are frozen

        if not (isinstance(getattr(model['args'], "soundbeam", None), dict) and model['args']['soundbeam']["apply"] and model['args']['soundbeam']["trainable"]):
            # not (soundbeam.apply and soundbeam.trainable)
            self.optimizer = initialize_config(self.optimizer_config)

        self.lr_scheduler_config = lr_scheduler

        if is_validation:
            self.validation_step = self._validation_step
            if metric:
                self.metric_config = metric
                self.metric_func = initialize_config(self.metric_config)
            else:
                self.metric_func = None
        
        self.is_validation = is_validation

    def forward(self, x):
        pass
    
    def set_train_mode(self):
        self.model.train()

    def training_step_processing(self, batch_data_dict, batch_idx):
        raise NotImplementedError

        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
            }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['ground_truth']}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def training_step(self, batch_data_dict, batch_idx):
        self.set_train_mode()

        batchsize, loss_dict = self.training_step_processing(batch_data_dict, batch_idx)

        loss = loss_dict['loss'] # for back propagation

        # log all items in loss_dict
        step_dict = {f'step_train/{name}': val.item() for name, val in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoc_dict = {f'epoch_train/{name}': val.item() for name, val in loss_dict.items()}
        self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)
        
        self.log_dict({"epoch/lr": self.optimizer.param_groups[0]['lr']},)  # this here results in a warning

        if isinstance(getattr(self, "soundbeam", None), dict) and self.soundbeam.get("apply", False):
            weights_dict = {f'weight_{i}': self.model.soundbeam_weights[i].item() for i in range(len(self.model.soundbeam_weights))}
            self.log_dict(weights_dict)

            if self.soundbeam['merge_method'] == 'add':
                if hasattr(self.model, "base") and hasattr(self.model.base, "weight_soundbeam_features"):
                    # weight_soundbeam_features, weight_x_center
                    self.log_dict({"weight_soundbeam_features": self.model.base.weight_soundbeam_features,
                                   "weight_x_center": self.model.base.weight_x_center
                                   })

        return loss


    def validation_step_processing(self, batch_data_dict, batch_idx):
        raise NotImplementedError
        batchsize = batch_data_dict['mixture'].shape[0]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_data_dict['label_vector'] # [bs, label_len]
            }
        output_dict = self.model(input_dict) # {'waveform': [bs, nch, wlen]}
        target_dict = {'waveform': batch_data_dict['ground_truth']}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict

    def _validation_step(self, batch_data_dict, batch_idx):
        self.model.eval()

        batchsize, loss_dict = self.validation_step_processing(batch_data_dict, batch_idx)

        # log all items in loss_dict
        step_dict = {f'step_val/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(step_dict, prog_bar=False, logger=True, on_epoch=False, on_step=True, sync_dist=True, batch_size=batchsize)
        epoc_dict = {f'epoch_val/{name}': metric for name, metric in loss_dict.items()}
        self.log_dict(epoc_dict, prog_bar=True, logger=True, on_epoch=True, on_step=False, sync_dist=True, batch_size=batchsize)

    def on_validation_epoch_end(self):
        raise NotImplementedError

    def test_step(self, batch_data):
        pass

    def on_test_epoch_end(self):
        raise NotImplementedError

    def configure_optimizers(self):
        r"""Configure optimizer.
            will be called automatically
        """
        if self.lr_scheduler_config and self.lr_scheduler_config['schedule_mode'] is not None:
            num_training_steps = self.trainer.estimated_stepping_batches

            schedule_mode = self.lr_scheduler_config['schedule_mode']
            num_warmup_steps = self.lr_scheduler_config['num_warmup_steps']
            lr_end = self.lr_scheduler_config['lr_end']

            print("schedule_mode: ", schedule_mode)
            print("num_warmup_steps: ", num_warmup_steps)
            print("lr_end: ", lr_end)

            scheduler = get_lr_scheduler(
                self.optimizer,
                num_training_steps,
                schedule_mode=schedule_mode,
                num_warmup_steps=num_warmup_steps,
                lr_end=lr_end
            )

            return {
                "optimizer": self.optimizer,
                "lr_scheduler": {
                    'scheduler': scheduler,
                    'interval': "step",
                    'frequency': 1,
                }
            }
        else:
            return self.optimizer
