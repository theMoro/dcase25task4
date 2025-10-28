"""Simplified S5 model for single tagger and separator."""
import torch
import torch.nn as nn

from src.utils import LABELS, initialize_config


class S5(torch.nn.Module):
    """Simplified S5 model with single tagger and separator."""
    
    def __init__(
        self,
        tagger_config,
        separator_config,
        label_set,
        tagger_ckpt=None,
        separator_ckpt=None,
        tagger_obj=None,
        soundbeam_config=None
    ):
        super().__init__()

        # -------- Tagger (single model) ----------------------------
        self.tagger = tagger_obj if tagger_obj is not None else initialize_config(tagger_config)
        self.tagger.eval()

        if tagger_ckpt is not None:
            self._load_ckpt(tagger_ckpt, self.tagger)

        # -------- Separator (single model) ------------------------
        self.separator = None
        if separator_config:
            # Add soundbeam config to separator args if provided
            if soundbeam_config is not None:
                separator_config["args"]["soundbeam"] = soundbeam_config
            
            self.separator = initialize_config(separator_config)
            self.separator.eval()
            
            if separator_ckpt is not None:
                self._load_ckpt(separator_ckpt, self.separator)

        self.label_set = label_set
        self.labels = LABELS[self.label_set]
        self.onehots = torch.eye(len(self.labels), requires_grad=False).to(torch.float32)
        self.label_onehots = {label: self.onehots[idx] for idx, label in enumerate(self.labels)}
        self.label_onehots['silence'] = torch.zeros(self.onehots.size(1), requires_grad=False, dtype=torch.float32)

    def _load_ckpt(self, path, model):
        """Load checkpoint with robust prefix handling."""
        model_ckpt = torch.load(path, weights_only=False, map_location='cpu')
        model_ckpt = model_ckpt['state_dict']
        
        if set(model.state_dict().keys()) != set(model_ckpt.keys()):
            # Remove prefix if checkpoint is from lightning module
            one_model_key = next(iter(model.state_dict().keys()))
            ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
            prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
            model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)}
        
        model.load_state_dict(model_ckpt)

    def _get_label(self, batch_multihot_vector):
        """Convert multihot vectors to label lists."""
        labels = []
        for multihot in batch_multihot_vector:
            label = [l for i, l in enumerate(self.labels) if multihot[i] > 0]
            labels.append(label)
        return labels

    def predict_label(self, waveforms, pthre=0.5, nevent_range=[1, 3]):
        """Predict labels from waveforms."""
        output = self.tagger.predict({'waveform': waveforms}, pthre=pthre, nevent_range=nevent_range)
        multihot = output['multihot_vector']

        def pad_silence(label_list):
            return label_list + ['silence'] * max(0, nevent_range[1] - len(label_list))

        if isinstance(multihot, torch.Tensor):
            # Single threshold behavior
            labels = self._get_label(multihot)
            labels = [pad_silence(l) for l in labels]
            out_dict = {'label': labels}
        elif isinstance(multihot, dict):
            # Multi-threshold behavior
            label_dict = {}
            for p, mh in multihot.items():
                labels = self._get_label(mh)
                labels = [pad_silence(l) for l in labels]
                label_dict[p] = labels
            out_dict = {'label': label_dict}
        else:
            raise TypeError(f"Unexpected type for multihot_vector: {type(multihot)}")

        # Pass through additional outputs
        if 'probabilities_strong' in output:
            out_dict['probabilities_strong'] = output['probabilities_strong']
        if 'logits_strong' in output:
            out_dict['logits_strong'] = output['logits_strong']

        return out_dict

    def _get_label_vector(self, batch_labels):
        """Convert batch labels to label vectors."""
        return torch.stack([
            torch.stack([self.label_onehots[label] for label in labels]).flatten() 
            for labels in batch_labels
        ])

    def separate(self, batch_mixture, batch_labels):
        """Separate sources given mixture and labels."""
        if self.separator is None:
            raise ValueError("No separator model loaded")
            
        label_vector = self._get_label_vector(batch_labels).to(batch_mixture.device)
        
        # Prepare input dict
        input_dict = {
            'mixture': batch_mixture,
            'label_vector': label_vector
        }
        
        # Add embeddings if separator uses soundbeam
        if hasattr(self.separator, 'embedding_model') and self.separator.embedding_model is not None:
            with torch.no_grad():
                out_dict = self.separator.embedding_model({'waveform': batch_mixture}, return_features=True)
                input_dict['embeddings'] = out_dict["features"]
                if "logits_strong" in out_dict.keys():
                    input_dict['labels_strong'] = out_dict["logits_strong"]
        
        # Forward pass
        with torch.no_grad():
            output_dict = self.separator(input_dict)
        
        return output_dict

    def predict_label_separate(self, mixture, pthre=0.5, nevent_range=[1, 3]):
        """Predict labels and separate sources in one step."""
        predict_labels = self.predict_label(mixture, pthre=pthre, nevent_range=nevent_range)
        predict_waveforms = self.separate(mixture, predict_labels['label'])
        
        return {
            'label': predict_labels['label'],
            'waveform': predict_waveforms['waveform']
        }