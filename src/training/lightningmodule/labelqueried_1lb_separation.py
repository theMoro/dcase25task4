import random
import torch
from src.training.lightningmodule.base_labelqueried_separation import BaseLabelQueriedSeparationModule


class LabelQueriedSeparationLightning1LB(BaseLabelQueriedSeparationModule):
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
        super().__init__(
            model=model,
            loss=loss,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            is_validation=is_validation,
            metric=metric,
            soundbeam=soundbeam,
            evaluation_tagger_model=evaluation_tagger_model
        )

        self.label_len = 18

    def _se_selection(self, batch_labels):
        return [
            random.choice([i for i in range(len(labels)) if labels[i]!='silence'])
            for labels in batch_labels
        ]

    def training_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]
        nsources = batch_data_dict['dry_sources'].shape[1]

        batch_labels = batch_data_dict['label']
        batch_gt = batch_data_dict['dry_sources'] # [bs, nsources, 1 ch, wlen]
        batch_label_vector = batch_data_dict['label_vector'] # [bs, nsources x nclasses]
        batch_label_vector_stack = batch_label_vector.view(batchsize, nsources, batch_label_vector.shape[-1]//nsources)# [bs, nsources, nclasses]

        # select a random source out of nsources sources
        selected_idx = self._se_selection(batch_labels)
        batch_gt_sel = torch.stack([batch_gt[i, se_idx, :, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, 1 ch, wlen]
        batch_label_vector_sel = torch.stack([batch_label_vector_stack[i, se_idx, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, nclasses]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen] = [5, 4, 320_000]
            'label_vector': batch_label_vector_sel # [bs, label_len]
        }

        if self.embedding_model:
            if self.embedding_model_is_trainable:
                out_dict = self.embedding_model({'waveform': batch_data_dict['mixture']}, return_features=True)
                input_dict['embeddings'] = out_dict["features"]

                if "logits_strong" in out_dict.keys():
                    input_dict['labels_strong'] = out_dict["logits_strong"]
            else:
                assert self.embedding_model.training == False, "Embedding model should be in eval mode"
                with torch.no_grad():
                    out_dict = self.embedding_model({'waveform': batch_data_dict['mixture']}, return_features=True)
                    input_dict['embeddings'] = out_dict["features"]

                    if "logits_strong" in out_dict.keys():
                        input_dict['labels_strong'] = out_dict["logits_strong"]

        output_dict = self.model(input_dict) # {'waveform': [bs, 1 ch, wlen]}
        target_dict = {'waveform': batch_gt_sel}
        loss_dict = self.loss_func(output_dict, target_dict)

        return batchsize, loss_dict

    def validation_step_processing(self, batch_data_dict, batch_idx):
        batchsize = batch_data_dict['mixture'].shape[0]
        nsources = batch_data_dict['dry_sources'].shape[1]

        batch_labels = batch_data_dict['label']
        batch_gt = batch_data_dict['dry_sources'] # [bs, nsources, 1 ch, wlen]
        batch_label_vector = batch_data_dict['label_vector'] # [bs, nsources x nclasses]
        batch_label_vector_stack = batch_label_vector.view(batchsize, nsources, batch_label_vector.shape[-1]//nsources)# [bs, nsources, nclasses]

        selected_idx = self._se_selection(batch_labels)
        batch_gt_sel = torch.stack([batch_gt[i, se_idx, :, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, 1 ch, wlen]
        batch_label_vector_sel = torch.stack([batch_label_vector_stack[i, se_idx, :] for i, se_idx in enumerate(selected_idx)], dim=0) # [bs, nclasses]

        input_dict = {
            'mixture': batch_data_dict['mixture'], # [bs, nch, wlen]
            'label_vector': batch_label_vector_sel # [bs, label_len]
        }

        if self.embedding_model:
            with torch.no_grad():
                out_dict = self.embedding_model({'waveform': batch_data_dict['mixture']}, return_features=True)
                input_dict['embeddings'] = out_dict["features"]

                if "logits_strong" in out_dict.keys():
                    input_dict['labels_strong'] = out_dict["logits_strong"]

        output_dict = self.model(input_dict) # {'waveform': [bs, 1 ch, wlen]}
        target_dict = {'waveform': batch_gt_sel}
        loss_dict = self.loss_func(output_dict, target_dict)

        loss_dict = {k: v.item() for k,v in loss_dict.items()}
        if self.metric_func: # add metrics
            metric = self.metric_func(output_dict, target_dict)
            for k,v in metric.items():
                loss_dict[k] = v.mean().item() # torch tensor size [bs]

        return batchsize, loss_dict
