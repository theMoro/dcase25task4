# copied from PretrainedSED repository
import os
from timm.models.layers import trunc_normal_

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import download_url_to_file
import torchaudio
from functools import partial
import numpy as np
import random
import wandb

from src.paths import RESOURCES_FOLDER, CHECKPOINT_URL
from src.models.pretrained_sed_models.m2d.M2D_wrapper import M2DWrapper
from src.paths import CKPT_PATH
from src.models.seq_models import BidirectionalLSTM, BidirectionalGRU

try:
    from src.paths import FINAL_CKPT_PATH
except ImportError:
    FINAL_CKPT_PATH = None


class Wrapper(nn.Module):
    """
        A wrapper module that adds an optional sequence model and classification heads on top of a transformer.

        Args:
            model_name (str): The name of the base model (transformer) providing sequence embeddings
            checkpoint (str, optional): checkpoint name for loading pre-trained weights. Default is None.
            ref_channel (int, optional): reference channel for transformer model. Default is 0.
            n_classes_strong (int): Number of classes for strong predictions. Default is 18.
            n_classes_weak (int, optional): Number of classes for weak predictions. Default is None,
                                            which sets it equal to n_classes_strong.
            embed_dim (int, optional): Embedding dimension of the base model output. Default is 768.
            seq_len (int, optional): Desired sequence length. Default is 250 (40 ms resolution).
            seq_model_type (str, optional): Type of sequence model to use.
                                            Default is None, which means no additional sequence model is used.
            head_type (str, optional): Type of classification head. Choices are ["linear", "attention", "None"].
                                       Default is "linear". "None" means that sequence embeddings are returned.
            rnn_layers (int, optional): Number of RNN layers if seq_model_type is "rnn". Default is 2.
            rnn_type (str, optional): Type of RNN to use. Choices are ["BiGRU", "BiLSTM"]. Default is "BiGRU".
            rnn_dim (int, optional): Dimension of RNN hidden state if seq_model_type is "rnn". Default is 256.
            rnn_dropout (float, optional): Dropout rate for RNN layers. Default is 0.0.
        """

    def __init__(self,
                 model_name,
                 checkpoint=None,
                 ref_channel=None,
                 data_sr=None,
                 target_sr=None,
                 freq_warp_p=0.0,
                 num_classes=None,
                 down_up_sample=False,
                 use_head_norm=True,
                 embed_dim=768,
                 seq_len=250,
                 seq_model_type=None,
                 head_type="linear",
                 rnn_layers=2,
                 rnn_type="BiGRU",
                 rnn_dim=2048,
                 rnn_dropout=0.0
                 ):
        super(Wrapper, self).__init__()

        self.model = None
        model_name = model_name.lower()
        self.model_name = model_name

        if checkpoint and checkpoint == "strong":
            checkpoint = "strong_1"

        if model_name == "m2d":
            assert isinstance(ref_channel, int)
            model = M2DWrapper()
            github_checkpoint = f"M2D_{checkpoint}" if checkpoint is not None else None
            embed_dim = 3840
        else:
            raise NotImplementedError(f"Model {model_name} not (yet) implemented")

        self.model = model
        self.ref_channel = ref_channel
        self.num_classes = num_classes
        self.down_up_sample = down_up_sample

        self.seq_len = seq_len
        self.embed_dim = embed_dim
        self.n_classes_strong = num_classes  # n_classes_strong
        self.n_classes_weak = num_classes  # n_classes_weak if n_classes_weak is not None else n_classes_strong
        self.seq_model_type = seq_model_type
        self.head_type = head_type

        self.freq_warp_p = freq_warp_p if freq_warp_p is not None else 0.0
        if self.freq_warp_p > 0:
            self.freq_warp = RandomResizeCrop((1, 1.0), time_scale=(1.0, 1.0))

        if data_sr is not None and target_sr is not None and data_sr != target_sr:
            print(f"Resampling audio for audio tagger from {data_sr} to {target_sr}")
            self.resample_fn = partial(
                torchaudio.functional.resample,
                orig_freq=data_sr,
                new_freq=target_sr,
                resampling_method="sinc_interp_kaiser",
            )
        else:
            self.resample_fn = None

        if self.seq_model_type == "rnn":
            if rnn_type == "BiGRU":
                self.seq_model = BidirectionalGRU(
                    n_in=self.embed_dim,
                    n_hidden=rnn_dim,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            elif rnn_type == "BiLSTM":
                self.seq_model = BidirectionalLSTM(
                    nIn=self.embed_dim,
                    nHidden=rnn_dim,
                    nOut=rnn_dim * 2,
                    dropout=rnn_dropout,
                    num_layers=rnn_layers
                )
            num_features = rnn_dim * 2
        elif self.seq_model_type is None:
            self.seq_model = nn.Identity()
            # no additional sequence model
            num_features = self.embed_dim
        else:
            raise ValueError(f"Unknown seq_model_type: {self.seq_model_type}")

        if self.head_type == "attention":
            self.use_head_norm = use_head_norm
            if use_head_norm:
                self.head_norm = torch.nn.BatchNorm1d(num_features, affine=False)

        if self.head_type in ["2_linear_layers", "attention"]:
            self.strong_head = nn.Linear(num_features, self.num_classes)
            self.weak_head = nn.Linear(num_features, self.num_classes)

        elif self.head_type == "linear":
            self.use_head_norm = use_head_norm
            if use_head_norm:
                self.head_norm = torch.nn.BatchNorm1d(num_features, affine=False)
            self.head = nn.Linear(num_features, self.num_classes)

            # from m2dat file
            trunc_normal_(self.head.weight, std=2e-5)

        if checkpoint is not None:
            print("Loading pretrained checkpoint: ", checkpoint)
            if any(checkpoint.endswith(s) for s in ["strong_1", "weak", "ssl"]):
                self.load_checkpoint(github_checkpoint)
            else:
                self.load_local_or_wandb_checkpoint(checkpoint)

    def load_local_or_wandb_checkpoint(self, checkpoint_id):
        """Locate and load a **.ckpt** file in the following order:

        1. **FINAL_CKPT_PATH / "tagger" / <wandb_id>.ckpt** – a consolidated export directory (optional).
        2. **CKPT_PATH** (legacy): sub‑folders named after the *wandb_id* containing exactly one .ckpt file.
        3. **Weights & Biases** artifact `cp_tobi/DCASE25_Task4/<wandb_id>:v0`.

        The first location that contains a matching checkpoint is used. The file's
        ``state_dict`` is stripped of the leading ``"model."`` prefix before being
        loaded into this wrapper.
        """

        found_path: str | None = None

        # --------------------------------------------------------------------
        # 1️⃣  FINAL_CKPT_PATH (if configured) ---------------------------------
        # --------------------------------------------------------------------
        if FINAL_CKPT_PATH:
            tagger_dir = os.path.join(FINAL_CKPT_PATH, "tagger")
            candidate = os.path.join(tagger_dir, f"{checkpoint_id}.ckpt")
            if os.path.exists(candidate):
                found_path = candidate
                print(f"Found checkpoint in FINAL_CKPT_PATH: {found_path}")

        # --------------------------------------------------------------------
        # 2️⃣  CKPT_PATH legacy structure --------------------------------------
        # --------------------------------------------------------------------
        if found_path is None:
            for root, _, files in os.walk(CKPT_PATH):
                # The wandb_id is the folder name; compare against leaf folder only.
                if checkpoint_id == os.path.basename(root):
                    ckpt_files = [f for f in files if f.endswith(".ckpt")]
                    if len(ckpt_files) == 1:
                        found_path = os.path.join(root, ckpt_files[0])
                    elif len(ckpt_files) > 1:
                        found_path = os.path.join(root, "last.ckpt")
                    if found_path:
                        print(f"Found checkpoint in CKPT_PATH: {found_path}")
                        break

        # --------------------------------------------------------------------
        # 3️⃣  Weights & Biases artifact --------------------------------------
        # --------------------------------------------------------------------
        if found_path is None:
            artifact_dir = f"artifacts/{checkpoint_id}:v0"
            artifact_spec = f"cp_tobi/DCASE25_Task4/{checkpoint_id}:v0"

            # Re‑use a previously downloaded artifact if present.
            if os.path.exists(artifact_dir):
                ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
                if len(ckpt_files) == 1:
                    found_path = os.path.join(artifact_dir, ckpt_files[0])
                    print(f"Found existing artifact at: {found_path}")
                elif len(ckpt_files) > 1:
                    raise ValueError(f"Multiple .ckpt files found in {artifact_dir}. Expected exactly one.")
            if found_path is None:
                try:
                    print(f"Downloading artifact {artifact_spec} from W&B …")
                    artifact = wandb.run.use_artifact(artifact_spec, type="model")
                    artifact_dir = artifact.download()
                    ckpt_files = [f for f in os.listdir(artifact_dir) if f.endswith(".ckpt")]
                    if len(ckpt_files) != 1:
                        raise FileNotFoundError(f"Expected one .ckpt file in {artifact_dir}, found {len(ckpt_files)}.")
                    found_path = os.path.join(artifact_dir, ckpt_files[0])
                except Exception as e:
                    print("Error downloading artifact:", e)

        # --------------------------------------------------------------------
        # Final checks & state_dict loading ----------------------------------
        # --------------------------------------------------------------------
        if found_path is None:
            raise FileNotFoundError(f"Checkpoint with ID '{checkpoint_id}' not found in FINAL_CKPT_PATH, "
                                    f"CKPT_PATH, or on Weights & Biases.")

        print(f"Loading checkpoint from: {found_path}")
        ckpt = torch.load(found_path, map_location="cpu")
        if "state_dict" not in ckpt:
            raise KeyError("Checkpoint does not contain a 'state_dict' key.")

        # Remove the leading "model." for compatibility with the wrapper.
        stripped_state_dict = {k.replace("model.", "", 1): v for k, v in ckpt["state_dict"].items() if k.startswith("model.")}
        self.load_state_dict(stripped_state_dict, strict=True)

    def load_checkpoint(self, checkpoint):
        ckpt_file = os.path.join(RESOURCES_FOLDER, checkpoint + ".pt")
        if not os.path.exists(ckpt_file):
            download_url_to_file(CHECKPOINT_URL + checkpoint + ".pt", ckpt_file)
        state_dict = torch.load(ckpt_file, map_location="cpu", weights_only=True)

        # compatibility with uniform wrapper structure we introduced for the public repo
        if 'fpasst' in checkpoint:
            state_dict = {("model.fpasst." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'M2D' in checkpoint:
            state_dict = {("model.m2d." + k[len("model."):] if not k.startswith("model.m2d.") and k.startswith("model.")
                           else k): v for k, v in state_dict.items()}
        elif 'BEATs' in checkpoint:
            state_dict = {("model.beats." + k[len("model.model."):] if k.startswith("model.model")
                           else k): v for k, v in state_dict.items()}
        elif 'ASIT' in checkpoint:
            state_dict = {("model.asit." + k[len("model."):] if k.startswith("model.")
                           else k): v for k, v in state_dict.items()}

        n_classes_weak_in_sd = state_dict['weak_head.bias'].shape[0] if 'weak_head.bias' in state_dict else -1
        n_classes_strong_in_sd = state_dict['strong_head.bias'].shape[0] if 'strong_head.bias' in state_dict else -1
        seq_model_in_sd = any(['seq_model.' in key for key in state_dict.keys()])
        keys_to_remove = []
        strict = True
        expected_missing = 0
        if self.head_type is None:
            # remove all keys related to head
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
        elif self.seq_model_type is not None and not seq_model_in_sd:
            # we want to train a sequence model (e.g., rnn) on top of a
            #   pre-trained transformer (e.g., AS weak pretrained)
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            num_seq_model_keys = len([key for key in self.seq_model.state_dict()])
            expected_missing = len(keys_to_remove) + num_seq_model_keys
            strict = False
        elif self.head_type == "linear":
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            if self.use_head_norm:
                expected_missing = 4
            else:
                expected_missing = 2
            strict = False
        elif self.head_type == "attention":
            keys_to_remove.append('weak_head.bias')
            keys_to_remove.append('weak_head.weight')
            keys_to_remove.append('strong_head.bias')
            keys_to_remove.append('strong_head.weight')
            if self.use_head_norm:
                expected_missing = 6
            else:
                expected_missing = 4
            strict = False
        else:
            # head type is not None
            if n_classes_weak_in_sd != self.n_classes_weak:
                # remove weak head from sd
                keys_to_remove.append('weak_head.bias')
                keys_to_remove.append('weak_head.weight')
                strict = False
            if n_classes_strong_in_sd != self.n_classes_strong:
                # remove strong head from sd
                keys_to_remove.append('strong_head.bias')
                keys_to_remove.append('strong_head.weight')
                strict = False
            expected_missing = len(keys_to_remove)

        # allow missing mel parameters for compatibility
        num_mel_keys = len([key for key in self.state_dict() if 'mel_transform' in key])
        if num_mel_keys > 0:
            expected_missing += num_mel_keys
            strict = False

        state_dict = {k: v for k, v in state_dict.items() if k not in keys_to_remove}
        if self.ref_channel == "all":
            if self.model_name == "fpasst":
                self.model.adapt_input_channels(state_dict, new_in_channels=4)
            elif self.model_name == "atst-f":
                self.model.adapt_input_channels(state_dict, new_in_channels=4)
            else:
                raise NotImplementedError(f"4 input channels not implemented for model {self.model_names}")
        missing, unexpected = self.load_state_dict(state_dict, strict=strict)
        assert len(missing) == expected_missing
        assert len(unexpected) == 0

    def separate_params(self, lr, lr_decay_factor, weight_decay):
        if hasattr(self, "separate_params"):
            pt_params = self.model.separate_params()
        else:
            raise NotImplementedError("The base model has no 'separate_params' method!'")

        seq_params = []
        head_params = []

        for name, p in self.named_parameters():
            if name.startswith('model'):
                # the transformer - already collected above
                pass
            elif name.startswith('seq_model'):
                # the optional sequence model
                seq_params.append(p)
            elif 'head' in name:
                # the prediction head
                head_params.append(p)
            else:
                raise ValueError(f"Unexpected key in model: {name}")

        param_groups = [
            {'params': head_params, 'lr': lr},  # model head (besides base model and seq model)
            {'params': seq_params, 'lr': lr},
        ]

        scale_lrs = [lr * (lr_decay_factor ** i) for i in range(1, len(pt_params) + 1)]

        param_groups = param_groups + [{"params": pt_params[i], "lr": scale_lrs[i]} for i in
                                       range(len(pt_params))]

        # do not apply weight decay to biases and batch norms
        param_groups_split = []
        for param_group in param_groups:
            params_1D, params_2D = [], []
            lr = param_group['lr']
            for param in param_group['params']:
                if param.ndimension() >= 2:
                    params_2D.append(param)
                elif param.ndimension() <= 1:
                    params_1D.append(param)
            param_groups_split += [{'params': params_2D, 'lr': lr, 'weight_decay': weight_decay},
                                   {'params': params_1D, 'lr': lr, 'weight_decay': 0.0}]

        return param_groups_split

    def has_separate_params(self):
        return hasattr(self.model, "separate_params")

    def mel_forward(self, x):
        if x.dim() == 2:
            # If input is (B, T), add a channel dimension -> (B, 1, T)
            x = x.unsqueeze(1)

        B, C, T = x.shape
        mel_per_channel = []

        for c in range(C):
            mel = self.model.mel_forward(x[:, c, :])  # shape (B, n_mels, time_frames)
            mel_per_channel.append(mel)

        mel = torch.cat(mel_per_channel, dim=1).squeeze(2)
        return mel

    def forward(self, input_dict, return_features=False):
        # (batch size x sequence length x embedding dimension)

        features = None
        batch_audio = input_dict['waveform']
        if batch_audio.dim() == 3:
            assert self.ref_channel is not None
            if isinstance(self.ref_channel, int):
                batch_audio = batch_audio[:, self.ref_channel:self.ref_channel + 1, :] # keep ref_channel dim
            else:
                assert self.ref_channel == "all"

        if self.resample_fn is not None:
            batch_audio = self.resample_fn(batch_audio)
        batch_audio = self.mel_forward(batch_audio)

        if self.freq_warp_p > 0 and torch.rand(1) < self.freq_warp_p:
            batch_audio = batch_audio.squeeze(1)
            batch_audio = self.freq_warp(batch_audio)
            batch_audio = batch_audio.unsqueeze(1)

        if return_features:
            x, features = self.model(batch_audio, return_features=True) # here they are already different
        else:
            x = self.model(batch_audio)

        # M2D: x.shape: batch size x 62 x 3840

        assert len(x.shape) == 3

        if self.down_up_sample:
            if x.size(-2) > self.seq_len:
                x = torch.nn.functional.adaptive_avg_pool1d(x.transpose(1, 2), self.seq_len).transpose(1, 2)
            elif x.size(-2) < self.seq_len:
                x = torch.nn.functional.interpolate(x.transpose(1, 2), size=self.seq_len,
                                                    mode='linear').transpose(1, 2)

        x = self.seq_model(x)

        if self.head_type == "attention":
            # attention head to obtain weak from strong predictions
            if self.use_head_norm:
                x = x.permute(0, 2, 1)
                x = self.head_norm(x)
                x = x.permute(0, 2, 1)  # back to [B, T, D]
            logits_strong = self.strong_head(x)
            strong = torch.sigmoid(logits_strong)
            logits_weak = self.weak_head(x)
            sof = torch.softmax(logits_weak, dim=-1)
            sof = torch.clamp(sof, min=1e-7, max=1)
            weak = (strong * sof).sum(1) / sof.sum(1)

            if return_features:
                output_dict = {'probabilities': weak, 'probabilities_strong': strong.transpose(1, 2),
                               'logits_strong': logits_strong.transpose(1, 2), 'logits_weak': logits_weak, 'features': features}
            else:
                output_dict = {'probabilities': weak, 'probabilities_strong': strong.transpose(1, 2),
                               'logits_strong': logits_strong.transpose(1, 2), 'logits_weak': logits_weak}

            return output_dict
        elif self.head_type == "2_linear_layers":
            strong = self.strong_head(x)
            weak = self.weak_head(x.mean(dim=1))
            return strong.transpose(1, 2), weak
        elif self.head_type == "linear":
            # here you take the mean of the features across the
            x = x.mean(1)  # B, D
            if self.use_head_norm:
                x = self.head_norm(x.unsqueeze(-1)).squeeze(-1)
            x = self.head(x)

            x = torch.sigmoid(x)

            if return_features:
                output_dict = {'features': features, 'probabilities': x}
            else:
                output_dict = {'probabilities': x}

            return output_dict
        else:
            # no head means the sequence is returned instead of strong and weak predictions
            return x

    def _get_top_prob(self, prob_vector, pthres, nevent_range):
        # prob_vector: [bs, nclasses]
        # pthres: list of thresholds
        batch_size, n_classes = prob_vector.shape
        all_onehots = {}  # dict: threshold → multihot [bs, nclasses]

        for pthre in pthres:
            onehots = torch.zeros_like(prob_vector)
            for i in range(batch_size):
                probs = prob_vector[i]
                selected_indices = (probs >= pthre).nonzero(as_tuple=True)[0]

                if len(selected_indices) < nevent_range[0]:
                    top_indices = torch.argsort(probs, descending=True)[:nevent_range[0]]
                else:
                    top_indices = selected_indices[
                        torch.argsort(probs[selected_indices], descending=True)[:nevent_range[1]]]
                onehots[i, top_indices] = 1.0
            all_onehots[pthre] = onehots
        return all_onehots

    def predict(self, input_dict, pthre=0.5, nevent_range=[1, 3]):
        output_dict = self.forward(input_dict)

        if isinstance(pthre, (float, int)):
            # Legacy behavior: return single multihot tensor
            output_dict['multihot_vector'] = self._get_top_prob(
                output_dict['probabilities'], [pthre], nevent_range)[pthre]
        else:
            # New behavior: return dict of multihot tensors per threshold
            output_dict['multihot_vector'] = self._get_top_prob(
                output_dict['probabilities'], pthre, nevent_range)

        return output_dict


class RandomResizeCrop(nn.Module):
    """Random Resize Crop block.

    Args:
        virtual_crop_scale: Virtual crop area `(F ratio, T ratio)` in ratio to input size.
        freq_scale: Random frequency range `(min, max)`.
        time_scale: Random time frame range `(min, max)`.
    """

    def __init__(self, virtual_crop_scale=(1.0, 1.5), freq_scale=(0.6, 1.0), time_scale=(0.6, 1.5)):
        super().__init__()
        self.virtual_crop_scale = virtual_crop_scale
        self.freq_scale = freq_scale
        self.time_scale = time_scale
        self.interpolation = 'bicubic'
        assert time_scale[1] >= 1.0 and freq_scale[1] >= 1.0

    @staticmethod
    def get_params(virtual_crop_size, in_size, time_scale, freq_scale):
        canvas_h, canvas_w = virtual_crop_size
        src_h, src_w = in_size
        h = np.clip(int(np.random.uniform(*freq_scale) * src_h), 1, canvas_h)
        w = np.clip(int(np.random.uniform(*time_scale) * src_w), 1, canvas_w)
        i = random.randint(0, canvas_h - h) if canvas_h > h else 0
        j = random.randint(0, canvas_w - w) if canvas_w > w else 0
        return i, j, h, w

    def forward(self, lms):
        # spec_output = []
        # for lms in specs:
        # lms = lms.unsqueeze(0)
        # make virtual_crop_arear empty space (virtual crop area) and copy the input log mel spectrogram to th the center
        virtual_crop_size = [int(s * c) for s, c in zip(lms.shape[-2:], self.virtual_crop_scale)]
        virtual_crop_area = (torch.zeros((lms.shape[0], virtual_crop_size[0], virtual_crop_size[1]))
                            .to(torch.float).to(lms.device))
        _, lh, lw = virtual_crop_area.shape
        c, h, w = lms.shape
        x, y = (lw - w) // 2, (lh - h) // 2
        virtual_crop_area[:, y:y+h, x:x+w] = lms
        # get random area
        i, j, h, w = self.get_params(virtual_crop_area.shape[-2:], lms.shape[-2:], self.time_scale, self.freq_scale)
        crop = virtual_crop_area[:, i:i+h, j:j+w]
        # print(f'shapes {virtual_crop_area.shape} {crop.shape} -> {lms.shape}')
        lms = F.interpolate(crop.unsqueeze(1), size=lms.shape[-2:],
            mode=self.interpolation, align_corners=True).squeeze(1)
            # spec_output.append(lms.float())
        return lms.float() # torch.concat(lms, dim=0)

    def __repr__(self):
        format_string = self.__class__.__name__ + f'(virtual_crop_size={self.virtual_crop_scale}'
        format_string += ', time_scale={0}'.format(tuple(round(s, 4) for s in self.time_scale))
        format_string += ', freq_scale={0})'.format(tuple(round(r, 4) for r in self.freq_scale))
        return format_string
