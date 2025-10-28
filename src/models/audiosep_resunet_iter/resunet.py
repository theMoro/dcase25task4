import numpy as np
from typing import Dict, List, NoReturn, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchlibrosa.stft import STFT, ISTFT, magphase
from .base import Base, init_layer, init_bn, act
import os
from .FaSNet import DPRNN
from torch.hub import download_url_to_file
import random

from src.paths import RESOURCES_FOLDER

AUDIOSEP_URL = "https://huggingface.co/spaces/Audio-AGI/AudioSep/resolve/main/checkpoint/"
CKPT_NAME = "audiosep_base_4M_steps.ckpt"


class FiLM(nn.Module):
    def __init__(self, film_meta, condition_size):
        super(FiLM, self).__init__()

        self.condition_size = condition_size

        self.modules, _ = self.create_film_modules(
            film_meta=film_meta,
            ancestor_names=[],
        )

    def create_film_modules(self, film_meta, ancestor_names):

        modules = {}

        # Pre-order traversal of modules
        for module_name, value in film_meta.items():

            if isinstance(value, int):

                ancestor_names.append(module_name)
                unique_module_name = '->'.join(ancestor_names)

                modules[module_name] = self.add_film_layer_to_module(
                    num_features=value,
                    unique_module_name=unique_module_name,
                )

            elif isinstance(value, dict):

                ancestor_names.append(module_name)

                modules[module_name], _ = self.create_film_modules(
                    film_meta=value,
                    ancestor_names=ancestor_names,
                )

            ancestor_names.pop()

        return modules, ancestor_names

    def add_film_layer_to_module(self, num_features, unique_module_name):

        layer = nn.Linear(self.condition_size, num_features)
        init_layer(layer)
        self.add_module(name=unique_module_name, module=layer)

        return layer

    def forward(self, conditions: torch.Tensor):
        """
        Parameters
        ----------
        conditions : (B, C_cond)          – legacy
                   | (B, T, C_cond)       – temporal
        """
        if conditions.dim() == 2:  # legacy path
            return self._calc_film(conditions, self.modules, has_time=False)

        if conditions.dim() == 3:  # temporal path
            B, T, C = conditions.shape
            if C != self.condition_size:
                raise ValueError(
                    f"FiLM: expected last dim == {self.condition_size}, "
                    f"got {C}."
                )
            return self._calc_film(conditions, self.modules, has_time=True)

        raise ValueError(
            "FiLM: 'conditions' must be 2-D (B,C) or 3-D (B,T,C); "
            f"got shape {tuple(conditions.shape)}."
        )

    # ------------------------------------------------------------------
    # recursive helper --------------------------------------------------
    # ------------------------------------------------------------------
    def _calc_film(self, conditions, modules, has_time: bool):
        """
        Recursively build a dictionary that mirrors `modules` but contains
        the β tensors produced by each leaf Linear layer.
        """
        film_data = {}
        for name, module in modules.items():
            if isinstance(module, nn.Module):  # leaf
                if not has_time:  # -------- legacy
                    beta = module(conditions)  # (B, C_feat)
                    beta = beta[:, :, None, None]  # (B, C_feat, 1, 1)
                else:  # -------- temporal
                    B, T, _ = conditions.shape
                    beta = module(conditions.view(B * T, -1)) \
                        .view(B, T, -1)  # (B, T, C_feat)
                    beta = beta.permute(0, 2, 1).unsqueeze(-1)
                    # shape: (B, C_feat, T, 1)
                film_data[name] = beta
            elif isinstance(module, dict):  # recurse
                film_data[name] = self._calc_film(
                    conditions, module, has_time
                )
        return film_data


class ConvBlockRes(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        momentum: float,
        has_film,
        time_film_mode: str = "additive"
    ):
        r"""Residual block."""
        super(ConvBlockRes, self).__init__()

        padding = [kernel_size[0] // 2, kernel_size[1] // 2]

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.bn2 = nn.BatchNorm2d(out_channels, momentum=momentum)

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, 1),
            dilation=(1, 1),
            padding=padding,
            bias=False,
        )

        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=(1, 1),
                stride=(1, 1),
                padding=(0, 0),
            )
            self.is_shortcut = True
        else:
            self.is_shortcut = False

        self.has_film = has_film
        self.time_film_mode = time_film_mode

        self.init_weights()

    def init_weights(self) -> NoReturn:
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_bn(self.bn2)
        init_layer(self.conv1)
        init_layer(self.conv2)

        if self.is_shortcut:
            init_layer(self.shortcut)

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict, time_film_dict=None) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        b1 = film_dict['beta1']
        b2 = film_dict['beta2']

        if time_film_dict is not None:
            time_b1 = time_film_dict['beta1']
            time_b2 = time_film_dict['beta2']
            time_dim = input_tensor.size(2)
            time_b1 = F.interpolate(time_b1, size=(time_dim, 1), mode='bilinear', align_corners=False)
            time_b2 = F.interpolate(time_b2, size=(time_dim, 1), mode='bilinear', align_corners=False)
        else:
            time_b1, time_b2 = 0, 0

        if self.time_film_mode == "additive" or isinstance(time_b1, int):
            a1 = self.bn1(input_tensor) + b1 + time_b1
        else:
            g1 = torch.sigmoid(time_b1)
            a1 = g1 * (self.bn1(input_tensor) + b1)

        x = self.conv1(F.leaky_relu_(a1, negative_slope=0.01))

        if self.time_film_mode == "additive" or isinstance(time_b2, int):
            a2 = self.bn2(x) + b2 + time_b2
        else:
            g2 = torch.sigmoid(time_b2)
            a2 = g2 * (self.bn2(x) + b2)

        x = self.conv2(F.leaky_relu_(a2, negative_slope=0.01))

        if self.is_shortcut:
            return self.shortcut(input_tensor) + x
        else:
            return input_tensor + x


class EncoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        downsample: Tuple,
        momentum: float,
        has_film,
        time_film_mode: str = "additive"
    ):
        r"""Encoder block, contains 8 convolutional layers."""
        super(EncoderBlockRes1B, self).__init__()

        self.conv_block1 = ConvBlockRes(
            in_channels, out_channels, kernel_size, momentum, has_film, time_film_mode
        )
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor, film_dict: Dict, time_film_dict=None) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            encoder_pool: (batch_size, output_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            encoder: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        if time_film_dict is None:
            time_film_dict = dict()

        encoder = self.conv_block1(input_tensor, film_dict['conv_block1'], time_film_dict.get('conv_block1', None))
        encoder_pool = F.avg_pool2d(encoder, kernel_size=self.downsample)
        return encoder_pool, encoder


class DecoderBlockRes1B(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Tuple,
        upsample: Tuple,
        momentum: float,
        has_film,
        time_film_mode: str = "additive"
    ):
        r"""Decoder block, contains 1 transposed convolutional and 8 convolutional layers."""
        super(DecoderBlockRes1B, self).__init__()
        self.kernel_size = kernel_size
        self.stride = upsample

        self.conv1 = torch.nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=self.stride,
            stride=self.stride,
            padding=(0, 0),
            bias=False,
            dilation=(1, 1),
        )

        self.bn1 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.conv_block2 = ConvBlockRes(
            out_channels * 2, out_channels, kernel_size, momentum, has_film, time_film_mode
        )
        self.bn2 = nn.BatchNorm2d(in_channels, momentum=momentum)
        self.has_film = has_film
        self.time_film_mode = time_film_mode

        self.init_weights()

    def init_weights(self):
        r"""Initialize weights."""
        init_bn(self.bn1)
        init_layer(self.conv1)

    def forward(
        self, input_tensor: torch.Tensor, concat_tensor: torch.Tensor, film_dict: Dict, time_film_dict=None
    ) -> torch.Tensor:
        r"""Forward data into the module.

        Args:
            input_tensor: (batch_size, input_feature_maps, downsampled_time_steps, downsampled_freq_bins)
            concat_tensor: (batch_size, input_feature_maps, time_steps, freq_bins)

        Returns:
            output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)
        """
        # b1 = film_dict['beta1']

        b1 = film_dict['beta1']

        if time_film_dict is not None:
            time_b1 = time_film_dict['beta1']
            time_dim = input_tensor.size(2)
            time_b1 = F.interpolate(time_b1, size=(time_dim, 1), mode='bilinear', align_corners=False)
        else:
            time_b1 = 0
            time_film_dict = dict()

        if self.time_film_mode == "additive" or isinstance(time_b1, int):
            a1 = self.bn1(input_tensor) + b1 + time_b1
        else:  # multiplicative – use sigmoid gate in [0,1]
            g1 = torch.sigmoid(time_b1)
            a1 = g1 * (self.bn1(input_tensor) + b1)

        x = self.conv1(F.leaky_relu_(a1))
        # (batch_size, input_feature_maps, time_steps, freq_bins)

        x = torch.cat((x, concat_tensor), dim=1)
        # (batch_size, input_feature_maps * 2, time_steps, freq_bins)

        x = self.conv_block2(x, film_dict['conv_block2'], time_film_dict.get('conv_block2', None))
        # output_tensor: (batch_size, output_feature_maps, time_steps, freq_bins)

        return x


class ResUNet30_Base(nn.Module, Base):
    def __init__(self, input_channels, output_channels, target_sources_num, dprnn, soundbeam=None,
                 channels_last=False, hop_size: int = 320, window_size: int = 2048,
                 time_film_mode: str = "additive", max_iterations=0):
        super(ResUNet30_Base, self).__init__()

        window_size = window_size
        hop_size = hop_size
        center = True
        pad_mode = "reflect"
        window = "hann"
        momentum = 0.01

        self.output_channels = output_channels
        self.orig_input_channels = input_channels
        self.total_in_channels = input_channels + 1  if max_iterations > 0 else input_channels# original mixture + one feedback
        self.target_sources_num = target_sources_num
        self.K = 3
        self.channels_last = channels_last
        self.max_iterations = max_iterations
        
        self.time_downsample_ratio = 2 ** 5  # This number equals 2^{#encoder_blcoks}

        self.stft = STFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.istft = ISTFT(
            n_fft=window_size,
            hop_length=hop_size,
            win_length=window_size,
            window=window,
            center=center,
            pad_mode=pad_mode,
            freeze_parameters=True,
        )

        self.bn0 = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)
        if self.max_iterations > 0:
            self.bn0_fb = nn.BatchNorm2d(window_size // 2 + 1, momentum=momentum)

        self.pre_conv = nn.Conv2d(
            in_channels=self.total_in_channels,
            out_channels=32, 
            kernel_size=(1, 1), 
            stride=(1, 1), 
            padding=(0, 0), 
            bias=True,
        )

        self.encoder_block1 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=32,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.encoder_block2 = EncoderBlockRes1B(
            in_channels=32,
            out_channels=64,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.encoder_block3 = EncoderBlockRes1B(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.encoder_block4 = EncoderBlockRes1B(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.encoder_block5 = EncoderBlockRes1B(
            in_channels=256,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.encoder_block6 = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.conv_block7a = EncoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            downsample=(1, 1),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block1 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(1, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block2 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=384,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block3 = DecoderBlockRes1B(
            in_channels=384,
            out_channels=256,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block4 = DecoderBlockRes1B(
            in_channels=256,
            out_channels=128,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block5 = DecoderBlockRes1B(
            in_channels=128,
            out_channels=64,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )
        self.decoder_block6 = DecoderBlockRes1B(
            in_channels=64,
            out_channels=32,
            kernel_size=(3, 3),
            upsample=(2, 2),
            momentum=momentum,
            has_film=True,
            time_film_mode=time_film_mode
        )

        self.after_conv = nn.Conv2d(
            in_channels=32,
            out_channels=input_channels * self.K * self.target_sources_num,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        self.out_conv = nn.Conv2d(
            in_channels=self.orig_input_channels,
            out_channels=output_channels,
            kernel_size=(1, 1),
            stride=(1, 1),
            padding=(0, 0),
            bias=True,
        )

        if soundbeam is None:
            soundbeam = {'apply': False}

        self.soundbeam = soundbeam

        if self.soundbeam['apply']:
            self.soundbeam_merge_method = self.soundbeam['merge_method']

            print(f"merge_method: {self.soundbeam_merge_method}")

            if self.soundbeam_merge_method == 'add':
                # Learnable weights for additive merging
                self.weight_x_center = nn.Parameter(torch.tensor(1.0))
                self.weight_soundbeam_features = nn.Parameter(torch.tensor(1.0))
            elif self.soundbeam_merge_method == 'concat':
                # Convolutional layer to reduce channels after concatenation
                self.merge_conv = nn.Conv2d(768, 384, kernel_size=1)
                self.merge_bn = nn.BatchNorm2d(768, momentum=momentum)

                # Initialize weights (e.g., Xavier init)
                nn.init.xavier_uniform_(self.merge_conv.weight)
                nn.init.zeros_(self.merge_conv.bias)
            else:
                raise ValueError("merge_method must be 'add' or 'concat'")

            if self.soundbeam['tagger_model']['args']['model_name'] == 'm2d':
                self.downsample_conv = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=1)
                self.downsample_bn = nn.BatchNorm2d(768, momentum=momentum)
            elif self.soundbeam['tagger_model']['args']['model_name'] == 'beats':
                self.downsample_conv = nn.Conv2d(in_channels=768, out_channels=384, kernel_size=1)
                self.downsample_bn = nn.BatchNorm2d(768, momentum=momentum)
            else:
                raise ValueError("Unknown tagger model name")

        self.init_weights()

        if dprnn['use']:
            self.dprnn = "post_sb" if dprnn['post_sb'] else "pre_sb"
            self.dprnn_dict = dprnn
        else:
            self.dprnn = None

        if self.dprnn:
            self.DPRNN = nn.Sequential(DPRNN(dprnn['type'], 384, dprnn['hidden'], 384,
                                             dprnn['dropout'], num_layers=dprnn['layers']))

    def init_weights(self):
        init_bn(self.bn0)
        init_layer(self.pre_conv)
        init_layer(self.after_conv)

    def feature_maps_to_wav(
        self,
        input_tensor: torch.Tensor,
        sp: torch.Tensor,
        sin_in: torch.Tensor,
        cos_in: torch.Tensor,
        audio_length: int,
        source_mask: torch.Tensor,
    ) -> torch.Tensor:
        r"""Convert feature maps to waveform.

        Args:
            input_tensor: (batch_size, target_sources_num * output_channels * self.K, time_steps, freq_bins)
            sp: (batch_size, input_channels, time_steps, freq_bins)
            sin_in: (batch_size, input_channels, time_steps, freq_bins)
            cos_in: (batch_size, input_channels, time_steps, freq_bins)

            (There is input_channels == output_channels for the source separation task.)

        Outputs:
            waveform: (batch_size, target_sources_num * output_channels, segment_samples)
        """
        batch_size, _, time_steps, freq_bins = input_tensor.shape

        x = input_tensor.reshape(
            batch_size,
            self.target_sources_num,
            self.orig_input_channels,
            self.K,
            time_steps,
            freq_bins,
        )
        # x: (batch_size, target_sources_num, output_channels, self.K, time_steps, freq_bins)

        mask_mag = torch.sigmoid(x[:, :, :, 0, :, :])
        _mask_real = torch.tanh(x[:, :, :, 1, :, :])
        _mask_imag = torch.tanh(x[:, :, :, 2, :, :])
        # linear_mag = torch.tanh(x[:, :, :, 3, :, :])
        _, mask_cos, mask_sin = magphase(_mask_real, _mask_imag)
        # mask_cos, mask_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Y = |Y|cos∠Y + j|Y|sin∠Y
        #   = |Y|cos(∠X + ∠M) + j|Y|sin(∠X + ∠M)
        #   = |Y|(cos∠X cos∠M - sin∠X sin∠M) + j|Y|(sin∠X cos∠M + cos∠X sin∠M)
        out_cos = (
            cos_in[:, None, :, :, :] * mask_cos - sin_in[:, None, :, :, :] * mask_sin
        )
        out_sin = (
            sin_in[:, None, :, :, :] * mask_cos + cos_in[:, None, :, :, :] * mask_sin
        )
        # out_cos: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)
        # out_sin: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate |Y|.
        out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag)
        # out_mag = F.relu_(sp[:, None, :, :, :] * mask_mag + linear_mag)
        # out_mag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Calculate Y_{real} and Y_{imag} for ISTFT.
        out_real = out_mag * out_cos
        out_imag = out_mag * out_sin
        # out_real, out_imag: (batch_size, target_sources_num, output_channels, time_steps, freq_bins)

        # Reformat shape to (N, 1, time_steps, freq_bins) for ISTFT where
        # N = batch_size * target_sources_num * input_channels

        shape = (batch_size * self.target_sources_num, self.orig_input_channels, time_steps, freq_bins)
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        out_real = self.out_conv(out_real)
        out_imag = self.out_conv(out_imag)
        # out_real_out_ch, out_real_out_ch: (batch_size * target_sources_num, output_channels, time_steps, freq_bins)

        shape = (
            batch_size * self.target_sources_num * self.output_channels,
            1,
            time_steps,
            freq_bins,
        )
        out_real = out_real.reshape(shape)
        out_imag = out_imag.reshape(shape)

        # ISTFT.
        x = self.istft(out_real, out_imag, audio_length)
        # (batch_size * target_sources_num * output_channels, segments_num)

        # Reshape.
        if self.target_sources_num == 1:
            waveform = x.reshape(
                batch_size, self.output_channels, audio_length
            )
            waveform = waveform * source_mask[:, :, None]
        else:
            waveform = x.reshape(
                batch_size, self.target_sources_num, self.output_channels, audio_length
            )
            waveform = waveform * source_mask[:, :, None, None]
        # (batch_size, target_sources_num * output_channels, segments_num)
        return waveform


    def forward(self, mixtures, film_dict, source_mask, weighted_avg=None, time_film_dict=None):
        """
        Args:
          input: (batch_size, segment_samples, channels_num)

        Outputs:
          output_dict: {
            'wav': (batch_size, segment_samples, channels_num),
            'sp': (batch_size, channels_num, time_steps, freq_bins)}
        """
        # (1) Split the “orig” mixture vs. the “feedback” channel
        mixture_orig_time = mixtures[:, :self.orig_input_channels, :]  # [B, orig_ch, L]
        feedback_time = mixtures[:, self.orig_input_channels:, :]  # [B, 1, L]

        # (2) Compute STFT only on the original mixture (no feedback)
        mag_orig, cos_in, sin_in = self.wav_to_spectrogram_phase(mixture_orig_time)
        # mag_orig, cos_in, sin_in: [B, orig_ch, T, F]

        if self.max_iterations > 0:
            # (3) Separately, compute a “spectrogram” of the feedback so the network can see it.
            mag_fb, _, _ = self.wav_to_spectrogram_phase(feedback_time)  # [B, 1, T, F]
            # (We discard cos/sin for feedback.)

            mag_orig_norm = self.bn0(mag_orig.transpose(1, 3)).transpose(1, 3)  # [B, 4, T, F]
            mag_fb_norm = self.bn0_fb(mag_fb.transpose(1, 3)).transpose(1, 3)  # [B, 1, T, F]
            mag_combined = torch.cat([mag_orig_norm, mag_fb_norm], dim=1)  # [B, 5, T, F]
            x = mag_combined
        else:
            x = self.bn0(mag_orig.transpose(1, 3)).transpose(1, 3)

        # Pad spectrogram to be evenly divided by downsample ratio.
        origin_len = x.shape[2]
        pad_len = (
            int(np.ceil(x.shape[2] / self.time_downsample_ratio)) * self.time_downsample_ratio
            - origin_len
        )
        x = F.pad(x, pad=(0, 0, 0, pad_len))
        """(batch_size, channels, padded_time_steps, freq_bins)"""

        # Let frequency bins be evenly divided by 2, e.g., 513 -> 512
        x = x[..., 0 : x.shape[-1] - 1]  # (bs, channels, T, F)

        if time_film_dict is None:
            time_film_dict = dict()

        # UNet
        x = self.pre_conv(x)
        x1_pool, x1 = self.encoder_block1(x, film_dict['encoder_block1'], time_film_dict.get('encoder_block1', None))  # x1_pool: (bs, 32, T / 2, F / 2)
        x2_pool, x2 = self.encoder_block2(x1_pool, film_dict['encoder_block2'], time_film_dict.get('encoder_block2', None))  # x2_pool: (bs, 64, T / 4, F / 4)
        x3_pool, x3 = self.encoder_block3(x2_pool, film_dict['encoder_block3'], time_film_dict.get('encoder_block3', None))  # x3_pool: (bs, 128, T / 8, F / 8)
        x4_pool, x4 = self.encoder_block4(x3_pool, film_dict['encoder_block4'], time_film_dict.get('encoder_block4', None))  # x4_pool: (bs, 256, T / 16, F / 16)
        x5_pool, x5 = self.encoder_block5(x4_pool, film_dict['encoder_block5'], time_film_dict.get('encoder_block5', None))  # x5_pool: (bs, 384, T / 32, F / 32)
        x6_pool, x6 = self.encoder_block6(x5_pool, film_dict['encoder_block6'], time_film_dict.get('encoder_block6', None))  # x6_pool: (bs, 384, T / 32, F / 64)
        x_center, _ = self.conv_block7a(x6_pool, film_dict['conv_block7a'], time_film_dict.get('conv_block7a', None))  # (bs, 384, T / 32, F / 64)

        # x_center.shape = (5, 384, 63, 8)
        # --> bs=5, 384 out-channels, 63 time-steps, 8 freq-bins

        # DPRNN Block before soundbeam
        if self.dprnn == "pre_sb":
            x_center = self.DPRNN(x_center)

        merged, features = None, None
        if self.soundbeam['apply']:
            if self.soundbeam['tagger_model']['args']['model_name'] == 'm2d':
                # weighted_avg: (bs, 310, 768)
                weighted_avg = weighted_avg.permute(0, 2, 1)  # (bs, 768, 310)
                weighted_avg = weighted_avg.unflatten(dim=2, sizes=(5, 62))  # (bs, 768, 5, 62)
                weighted_avg = weighted_avg.permute(0, 1, 3, 2)  # (bs, 768, 62, 5)

                features = F.leaky_relu_(self.downsample_bn(weighted_avg), negative_slope=0.01)

                # use interpolate to downsample
                features = F.interpolate(features, size=(x_center.shape[2], x_center.shape[3]), mode='bilinear',
                                         align_corners=False)  # (bs, 768, 63, 8)

                features = self.downsample_conv(features)
            elif self.soundbeam['tagger_model']['args']['model_name'] == 'beats':
                # weighted_avg: (bs, 496, 768)
                weighted_avg = weighted_avg.permute(0, 2, 1).contiguous()  # (bs, 768, 496)
                weighted_avg = weighted_avg.unflatten(dim=2, sizes=(62, 8))  # (bs, 768, 62, 8)

                features = F.leaky_relu_(self.downsample_bn(weighted_avg), negative_slope=0.01)

                # # pad features to match x_center
                # pad_len = (0, 0, 0, x_center.shape[2] - features.shape[2])
                # features = F.pad(features, pad=pad_len)  # (bs, 768, 63, 8)

                # use interpolate to downsample
                features = F.interpolate(features, size=(x_center.shape[2], x_center.shape[3]), mode='bilinear',
                                         align_corners=False)  # (bs, 768, 63, 8)

                features = self.downsample_conv(features)
            else:
                raise ValueError("Unknown tagger model name")

            if self.soundbeam_merge_method == 'add':
                # Normalize both tensors
                x_center_mean = x_center.mean(dim=[2, 3], keepdim=True)
                x_center_std = x_center.std(dim=[2, 3], keepdim=True) + 1e-6
                m2d_mean = features.mean(dim=[2, 3], keepdim=True)
                m2d_std = features.std(dim=[2, 3], keepdim=True) + 1e-6

                x_center_norm = (x_center - x_center_mean) / x_center_std
                m2d_norm = (features - m2d_mean) / m2d_std

                # Weighted addition
                merged = self.weight_x_center * x_center_norm + self.weight_soundbeam_features * m2d_norm
            elif self.soundbeam_merge_method == 'concat':
                # Concatenate along channel dimension
                concatenated = torch.cat([x_center, features], dim=1)  # (bs, 768, 63, 8)
                merged = self.merge_conv(
                    F.leaky_relu_(self.merge_bn(concatenated), negative_slope=0.01))  # (bs, 384, 63, 8)
        else:
            merged = x_center

        # DPRNN Block before soundbeam
        if self.dprnn == "post_sb":
            merged = self.DPRNN(merged)

        x7 = self.decoder_block1(merged, x6, film_dict['decoder_block1'],
                                 time_film_dict.get('decoder_block1', None))  # (bs, 384, T / 32, F / 32)
        x8 = self.decoder_block2(x7, x5, film_dict['decoder_block2'],
                                 time_film_dict.get('decoder_block2', None))  # (bs, 384, T / 16, F / 16)
        x9 = self.decoder_block3(x8, x4, film_dict['decoder_block3'],
                                 time_film_dict.get('decoder_block3', None))  # (bs, 256, T / 8, F / 8)
        x10 = self.decoder_block4(x9, x3, film_dict['decoder_block4'],
                                  time_film_dict.get('decoder_block4', None))  # (bs, 128, T / 4, F / 4)
        x11 = self.decoder_block5(x10, x2, film_dict['decoder_block5'],
                                  time_film_dict.get('decoder_block5', None))  # (bs, 64, T / 2, F / 2)
        x12 = self.decoder_block6(x11, x1, film_dict['decoder_block6'],
                                  time_film_dict.get('decoder_block6', None))  # (bs, 32, T, F)

        x = self.after_conv(x12)

        # Recover shape
        x = F.pad(x, pad=(0, 1))
        x = x[:, :, 0:origin_len, :]

        audio_length = mixtures.shape[2]

        # Recover each subband spectrograms to subband waveforms. Then synthesis
        # the subband waveforms to a waveform.
        separated_audio = self.feature_maps_to_wav(
            input_tensor=x,
            # input_tensor: (batch_size, target_sources_num * output_channels * self.K, T, F')
            sp=mag_orig,
            # sp: (batch_size, input_channels, T, F')
            sin_in=sin_in,
            # sin_in: (batch_size, input_channels, T, F')
            cos_in=cos_in,
            # cos_in: (batch_size, input_channels, T, F')
            audio_length=audio_length,
            source_mask=source_mask
        )
        # （batch_size, target_sources_num * output_channels, subbands_num, segment_samples)

        output_dict = {'waveform': separated_audio, 'source_mask': source_mask}

        return output_dict


def get_film_meta(module):

    film_meta = {}

    if hasattr(module, 'has_film'):\

        if module.has_film:
            film_meta['beta1'] = module.bn1.num_features
            film_meta['beta2'] = module.bn2.num_features
        else:
            film_meta['beta1'] = 0
            film_meta['beta2'] = 0

    for child_name, child_module in module.named_children():

        child_meta = get_film_meta(child_module)

        if len(child_meta) > 0:
            film_meta[child_name] = child_meta
    
    return film_meta


class IterativeSeparator(nn.Module):
    """
    Wraps a 5-channel ResUNet so that
      • ch1–4 : original mixture channels
      • ch5   : feedback (silent on first pass)
    If `max_iterations==1` the wrapper performs ONE forward pass (legacy).
    """
    def __init__(self, ss_model: ResUNet30_Base, max_iterations: int = 3, detach_feedback: bool = True):
        super().__init__()
        assert max_iterations >= 1, "`max_iterations` must be ≥1"
        self.ss_model       = ss_model
        self.max_iterations = max_iterations
        self.detach_feedback = detach_feedback

    def _sample_n_iter(self):
        if not torch.distributed.is_initialized():
            return torch.randint(1, self.max_iterations + 1, (1,)).item()

        # sample on rank-0, broadcast to all
        if torch.distributed.get_rank() == 0:
            n_iter = torch.randint(
                1, self.max_iterations + 1, (1,), device='cuda'
            )
        else:
            n_iter = torch.empty(1, dtype=torch.long, device='cuda')

        torch.distributed.broadcast(n_iter, src=0)
        return n_iter.item()

    def forward(self, mixtures, film_dict, source_mask, weighted_avg=None, time_film_dict=None, return_fb_list=False):
        """
        • Training  : pass an explicit `n_iter` sampled in [1 … max_iterations]
        • Inference : leave `n_iter=None`  → runs full `max_iterations`
        """
        B, C4, L = mixtures.shape
        assert C4 == 4, "Expecting 4-channel mixture"

        # decide how many unrolled steps
        if self.training:
            n_iter = self._sample_n_iter()
        else:
            n_iter = self.max_iterations

        fb = torch.zeros(B, 1, L, device=mixtures.device, dtype=mixtures.dtype)  # silent ch-5
        fb_list = []
        out_dict = {}

        for i in range(n_iter):
            mix_5ch = torch.cat([mixtures, fb], dim=1)          # [B, 5, L]

            with torch.set_grad_enabled(((i == n_iter-1) | (not self.detach_feedback)) & self.training):
                out_dict = self.ss_model(
                    mixtures=mix_5ch,
                    film_dict=film_dict,
                    source_mask=source_mask,  # [bs, target_sources_num]
                    weighted_avg=weighted_avg,
                    time_film_dict=time_film_dict
                )                      # core forward
                fb       = out_dict["waveform"]
                if self.detach_feedback:
                    fb = fb.detach()  # detach!

                fb_list.append(fb.squeeze(1))  # [B, L] (remove ch-5 dim)

        # out_dict already holds 'waveform' & 'source_mask'
        out_dict["n_iter"] = n_iter

        if return_fb_list:
            return out_dict, fb_list

        return out_dict


class ResUNet30(nn.Module):
    def __init__(self, input_channels, output_channels, condition_size, target_sources_num,
                 label_len, dprnn, soundbeam=None, use_time_film=False, time_film_mode: str = "additive",
                 binarize_sed_preds: bool = False, channels_last=False, load_pretrained=True,
                 hop_size: int = 320, window_size: int = 2048, max_iterations: int = 3,
                 detach_feedback: bool = True):
        super(ResUNet30, self).__init__()

        print("audiosep_resunet_iter.resunet.py: ResUNet30.__init__()")

        self.target_sources_num = target_sources_num
        self.max_iterations = max_iterations

        self.time_film_mode = time_film_mode.lower()
        if self.time_film_mode not in ("additive", "multiplicative"):
            raise ValueError("'time_film_mode' must be 'additive' or 'multiplicative'")

        self.base = ResUNet30_Base(
            input_channels= input_channels,
            output_channels=output_channels,
            target_sources_num=target_sources_num,
            dprnn=dprnn,
            soundbeam=soundbeam,
            channels_last=channels_last,
            hop_size=hop_size,
            window_size=window_size,
            time_film_mode=self.time_film_mode,
            max_iterations=self.max_iterations,
        )

        if channels_last:
            self.base = self.base.to(memory_format=torch.channels_last)
        
        self.film_meta = get_film_meta(
            module=self.base,
        )

        # add label embedding network for one-hot vectors
        self.label_embedding = nn.Sequential(
            nn.Linear(label_len, condition_size),
            nn.LayerNorm(condition_size),
            nn.ReLU(),
            nn.Linear(condition_size, condition_size),
            nn.LayerNorm(condition_size),
            nn.ReLU())
        
        self.film = FiLM(
            film_meta=self.film_meta, 
            condition_size=condition_size
        )

        if load_pretrained:
            # Ensure resources directory exists
            os.makedirs(RESOURCES_FOLDER, exist_ok=True)
            ckpt_file = os.path.join(RESOURCES_FOLDER, CKPT_NAME)

            # Download checkpoint if missing
            if not os.path.exists(ckpt_file):
                print(f"Downloading checkpoint from {AUDIOSEP_URL + CKPT_NAME}")
                download_url_to_file(f" {AUDIOSEP_URL + CKPT_NAME}", ckpt_file)

            ckpt = torch.load(ckpt_file, map_location='cpu')

            state_dict = {
                k[len("ss_model."):]: v
                for k, v in ckpt["state_dict"].items()
                if k.startswith("ss_model.base.") or k.startswith("ss_model.film.")
            }

            # Adjust pre_conv weight if shape mismatch: [32, 1, 1, 1] → [32, 4, 1, 1]
            pre_conv_key = 'base.pre_conv.weight'

            if pre_conv_key in state_dict and state_dict[pre_conv_key].shape[1] == 1:
                print(f"Patching {pre_conv_key} for 4 input channels")
                # replicate 4×, average, **append a zero kernel** for ch-5 (so legacy ≡ ch1–4 avg)
                w1 = state_dict[pre_conv_key]
                w = w1.repeat(1, 4, 1, 1) / 4.0  # [32,4,1,1]
                if self.max_iterations > 0:
                    z = torch.zeros_like(w1)  # [32,1,1,1]  (for silent ch-5)
                    w = (torch.cat([w, z], dim=1))
                state_dict[pre_conv_key] = w


            after_conv_weight_key = "base.after_conv.weight"
            after_conv_bias_key = "base.after_conv.bias"

            if self.target_sources_num >= 1:
                if after_conv_weight_key in state_dict and after_conv_bias_key in state_dict:
                    print(f"Patching {after_conv_weight_key} and {after_conv_bias_key} "
                          f"for {self.target_sources_num} sources.")
                    state_dict[after_conv_weight_key] = (
                        state_dict[after_conv_weight_key].repeat(self.target_sources_num * input_channels, 1, 1, 1))
                    state_dict[after_conv_bias_key] = (
                        state_dict[after_conv_bias_key].repeat(self.target_sources_num * input_channels))

            missing, unexpected = self.load_state_dict(state_dict, strict=False)
            assert all([k.startswith("base.out_conv.") or k.startswith("label_embedding.")
                        or k.startswith("base.downsample_") or k.startswith("base.weight_") or
                        k.startswith("base.bn0_fb") or k.startswith("base.DPRNN") for k in missing ])
            assert len(unexpected) == 0

        self.use_time_film = use_time_film
        if self.use_time_film:
            self.time_embedding = nn.Sequential(
                nn.Linear(target_sources_num, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 512),
                nn.LayerNorm(512),
                nn.ReLU())

            self.time_film = FiLM(
                film_meta=self.film_meta,
                condition_size=512
            )

        self.binarize_sed_preds = binarize_sed_preds

        if soundbeam is None:
            soundbeam = {'apply': False}

        self.soundbeam = soundbeam

        if self.soundbeam['apply']:
            if self.soundbeam['tagger_model']['args']['model_name'] == 'm2d':
                self.soundbeam_weights = nn.Parameter(torch.ones(13) / 13)  # learnable weights
            elif self.soundbeam['tagger_model']['args']['model_name'] == 'beats':
                self.soundbeam_weights = nn.Parameter(torch.ones(12) / 12)  # learnable weights
            else:
                raise ValueError("Unknown tagger model name")

        if max_iterations > 0:
            self.base = IterativeSeparator(
                self.base,
                max_iterations,
                detach_feedback
            )


    def forward(self, input_dict):
        mixtures = input_dict['mixture']
        label_vector = input_dict['label_vector']
        embeddings = input_dict['embeddings'] if 'embeddings' in input_dict else None
        labels_strong = input_dict['labels_strong'] if 'labels_strong' in input_dict and self.use_time_film else None

        if labels_strong is not None and self.use_time_film:
            B, C, T = labels_strong.shape
            if self.binarize_sed_preds:
                # threshold logits at 0 → binary on/off
                labels_strong = (labels_strong > 0).to(labels_strong.dtype)

            # Reshape label_vector to [B, 3, C]
            label_vector_reshaped = label_vector.view(B, self.target_sources_num, C)
            # Argmax over class axis to get indices — still safe even if all-zero
            indices = label_vector_reshaped.argmax(dim=-1)  # shape: [B, 3] for resunetk, [B, 1] for resunet

            # Check for silent entries (all zeros → max == 0 and sum == 0)
            is_silent = (label_vector_reshaped.sum(dim=-1) == 0)  # shape: [B, 3], bool

            # Use batch-wise indexing to extract the selected class predictions
            labels_strong_selected = torch.stack([
                labels_strong[b, indices[b]]  # [3, T]
                for b in range(B)
            ], dim=0)  # shape: [B, 3, T]

            # Set silent entries to zero
            labels_strong_selected[is_silent] = 0.0

            time_conditions = self.time_embedding(labels_strong_selected.reshape(B * T, -1))
            time_conditions = time_conditions.view(B, T, -1)

            time_film_dict = self.film(
                conditions=time_conditions,
            )
        else:
            time_film_dict = None

        with torch.no_grad():
            copylb = label_vector.clone()
            bs, lbl = copylb.shape
            source_mask = copylb.view(bs, self.target_sources_num, lbl // self.target_sources_num).sum(dim=2)
            source_mask = (source_mask != 0).to(label_vector.dtype)
        conditions = self.label_embedding(label_vector)
        film_dict = self.film(
            conditions=conditions,
        )

        # Compute weighted average of M2D embeddings if provided
        weighted_avg = None
        if self.soundbeam['apply']:
            weights = F.softmax(self.soundbeam_weights, dim=0)
            stacked_embeddings = torch.stack(embeddings, dim=1)  # (bs, n_blocks, 310, 768)
            weighted_avg = torch.sum(stacked_embeddings * weights[None, :, None, None], dim=1)  # (bs, 310, 768)

        output_dict = self.base(  # forward function
            mixtures=mixtures,
            film_dict=film_dict,
            source_mask=source_mask,  # [bs, target_sources_num]
            weighted_avg=weighted_avg,
            time_film_dict=time_film_dict
        )

        return output_dict


    def forward_inference(self, input_dict, nsources=3, label_len=18, return_fb_list=False):
        with torch.no_grad(): # this model is just for inference
            bs, lblen = input_dict['label_vector'].shape # [bs, label_len x nsources]

            assert (lblen == label_len * nsources)
            label_vector = input_dict['label_vector']

            embeddings = input_dict['embeddings'] if 'embeddings' in input_dict else None
            labels_strong = input_dict[
                'labels_strong'] if 'labels_strong' in input_dict and self.use_time_film else None

            if labels_strong is not None and self.use_time_film:
                B, C, T = labels_strong.shape
                if self.binarize_sed_preds:
                    labels_strong = (labels_strong > 0).to(labels_strong.dtype)
                # Reshape label_vector to [B, 3, C]
                label_vector = label_vector.view(B, nsources, C)  # self.target_sources_num = 3 (ResUNetK), 1 (ResUNet)  --> therefore use nsources
                label_vector_reshaped = label_vector.reshape(B * nsources, 1, label_len)
                # Argmax over class axis to get indices — still safe even if all-zero
                indices = label_vector_reshaped.argmax(dim=-1)  # shape: [B * nsources, 1]

                # Check for silent entries (all zeros → max == 0 and sum == 0)
                is_silent = (label_vector_reshaped.sum(dim=-1) == 0)  # shape: [B * nsources, 1], bool

                labels_strong = labels_strong.repeat_interleave(nsources, dim=0) # shape: [B * nsources, 18]

                # Use batch-wise indexing to extract the selected class predictions
                labels_strong_selected = torch.stack([
                    labels_strong[b, indices[b]]
                    for b in range(B * nsources)
                ], dim=0)  # shape: [B * nsources, 1, T]

                # Set silent entries to zero
                labels_strong_selected[is_silent] = 0.0

                time_conditions = self.time_embedding(labels_strong_selected.reshape(B * nsources * T, -1))
                time_conditions = time_conditions.view(B * nsources, T, -1)

                time_film_dict = self.film(
                    conditions=time_conditions,
                )
            else:
                time_film_dict = None

            with torch.no_grad():
                copylb = label_vector.clone()
                source_mask = copylb.view(bs, nsources, label_len).sum(dim=2)
                source_mask = (source_mask!=0).to(label_vector.dtype)
                source_mask = source_mask.reshape(bs * nsources, 1)
            label_vector = label_vector.reshape(bs, nsources, label_len)
            label_vector = label_vector.reshape(bs * nsources, label_len)
            mixtures = input_dict['mixture']
            mixtures_repeat = mixtures.repeat_interleave(nsources, dim=0)
            conditions = self.label_embedding(label_vector)
            film_dict = self.film(
                conditions=conditions,
            )

            weighted_avg = None
            if self.soundbeam['apply']:
                weights = F.softmax(self.soundbeam_weights, dim=0)
                stacked_embeddings = torch.stack(embeddings, dim=1)
                weighted_avg = torch.sum(stacked_embeddings * weights[None, :, None, None], dim=1)

                # extra step here:
                weighted_avg = weighted_avg.repeat_interleave(nsources, dim=0)

            output_dict_ = self.base(
                mixtures=mixtures_repeat,
                film_dict=film_dict,
                source_mask=source_mask,
                weighted_avg=weighted_avg,
                time_film_dict=time_film_dict,
                return_fb_list=return_fb_list
            )

            if return_fb_list:
                output_dict, fb_list = output_dict_
            else:
                output_dict = output_dict_

            # return 0 if label is silence
            # output_dict['waveform'] = output_dict['waveform']*label_vector.sum(1)[:, None, None]
            output_dict['waveform'] = output_dict['waveform'].reshape(bs, nsources, 1, mixtures.shape[-1])
            output_dict['source_mask'] = output_dict['source_mask'].reshape(bs, nsources, 1)

        if return_fb_list:
            return output_dict, fb_list

        return output_dict


    @torch.no_grad()
    def chunk_inference(self, input_dict):
        chunk_config = {
                    'NL': 1.0,
                    'NC': 3.0,
                    'NR': 1.0,
                    'RATE': 32000
                }

        mixtures = input_dict['mixture']
        conditions = input_dict['condition']

        film_dict = self.film(
            conditions=conditions,
        )

        NL = int(chunk_config['NL'] * chunk_config['RATE'])
        NC = int(chunk_config['NC'] * chunk_config['RATE'])
        NR = int(chunk_config['NR'] * chunk_config['RATE'])

        L = mixtures.shape[2]
        
        out_np = np.zeros([1, L])

        WINDOW = NL + NC + NR
        current_idx = 0

        while current_idx + WINDOW < L:
            chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]

            chunk_out = self.base(
                mixtures=chunk_in, 
                film_dict=film_dict,
            )['waveform']
            
            chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

            if current_idx == 0:
                out_np[:, current_idx:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, :-NR] if NR != 0 else chunk_out_np
            else:
                out_np[:, current_idx+NL:current_idx+WINDOW-NR] = \
                    chunk_out_np[:, NL:-NR] if NR != 0 else chunk_out_np[:, NL:]

            current_idx += NC

            if current_idx < L:
                chunk_in = mixtures[:, :, current_idx:current_idx + WINDOW]
                chunk_out = self.base(
                    mixtures=chunk_in, 
                    film_dict=film_dict,
                )['waveform']

                chunk_out_np = chunk_out.squeeze(0).cpu().data.numpy()

                seg_len = chunk_out_np.shape[1]
                out_np[:, current_idx + NL:current_idx + seg_len] = \
                    chunk_out_np[:, NL:]

        return out_np

    def parameter_groups(self, lr_pretrained: float, lr_new: float,
                         lr_dprnn: float, weight_decay: float):
        pretrained, new, dprnn = [], [], []
        covered = set()

        for name, param in self.named_parameters():
            if any([
                name.startswith("base.ss_model.encoder_block"),
                name.startswith("base.ss_model.decoder_block"),
                name.startswith("base.ss_model.conv_block7a"),
                name.startswith("base.ss_model.film"),
                name.startswith("film."),
            ]):
                pretrained.append((name, param))
                covered.add(name)

            elif any([
                name.startswith("base.ss_model.stft"),
                name.startswith("base.ss_model.istft"),
                name.startswith("base.ss_model.bn0"),
                name.startswith("base.ss_model.bn0_fb"),
                name.startswith("label_embedding"),
                name.startswith("time_embedding"),
                name.startswith("time_film."),
                name.startswith("base.ss_model.out_conv"),
                name.startswith("base.ss_model.weight_"),
                name.startswith("base.ss_model.downsample_"),
                name.startswith("base.ss_model.after_conv"),
                name.startswith("base.ss_model.pre_conv"),
                name.startswith("soundbeam_weights"),
            ]):
                new.append((name, param))
                covered.add(name)
            elif name.startswith("base.ss_model.DPRNN"):
                dprnn.append((name, param))
                covered.add(name)

        # Safety check
        all_params = set(n for n, _ in self.named_parameters())
        uncovered = all_params - covered
        if uncovered:
            raise ValueError(f"Uncovered parameters in split: {sorted(uncovered)}")

        def group_by_decay(params: List[Tuple[str, torch.nn.Parameter]], lr: float):
            with_decay, no_decay = [], []
            for name, p in params:
                if p.ndimension() >= 2:
                    with_decay.append(p)
                else:
                    no_decay.append(p)
            return [
                {'params': with_decay, 'lr': lr, 'weight_decay': weight_decay},
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
            ]

        param_groups = (group_by_decay(pretrained, lr_pretrained) + group_by_decay(new, lr_new)
                        + group_by_decay(dprnn, lr_dprnn))
        return param_groups

    def parameter_groups_without_iter(self, lr_pretrained: float, lr_new: float,
                         lr_dprnn: float, weight_decay: float):
        pretrained, new, dprnn = [], [], []
        covered = set()

        for name, param in self.named_parameters():
            if any([
                name.startswith("base.encoder_block"),
                name.startswith("base.decoder_block"),
                name.startswith("base.conv_block7a"),
                name.startswith("base.film"),
                name.startswith("film."),
            ]):
                pretrained.append((name, param))
                covered.add(name)

            elif any([
                name.startswith("base.stft"),
                name.startswith("base.istft"),
                name.startswith("base.bn0"),
                name.startswith("label_embedding"),
                name.startswith("time_embedding"),
                name.startswith("time_film."),
                name.startswith("base.out_conv"),
                name.startswith("base.weight_"),
                name.startswith("base.downsample_"),
                name.startswith("base.after_conv"),
                name.startswith("base.pre_conv"),
                name.startswith("soundbeam_weights"),
            ]):
                new.append((name, param))
                covered.add(name)
            elif name.startswith("base.DPRNN"):
                dprnn.append((name, param))
                covered.add(name)

        # Safety check
        all_params = set(n for n, _ in self.named_parameters())
        uncovered = all_params - covered
        if uncovered:
            raise ValueError(f"Uncovered parameters in split: {sorted(uncovered)}")

        def group_by_decay(params: List[Tuple[str, torch.nn.Parameter]], lr: float):
            with_decay, no_decay = [], []
            for name, p in params:
                if p.ndimension() >= 2:
                    with_decay.append(p)
                else:
                    no_decay.append(p)
            return [
                {'params': with_decay, 'lr': lr, 'weight_decay': weight_decay},
                {'params': no_decay, 'lr': lr, 'weight_decay': 0.0},
            ]

        param_groups = (group_by_decay(pretrained, lr_pretrained) + group_by_decay(new, lr_new)
                        + group_by_decay(dprnn, lr_dprnn))
        return param_groups

    def separate_params(self, lr: float, lr_dprnn: float, lr_decay_factor: float, weight_decay: float):
        if self.max_iterations > 0:
            return self.parameter_groups(
                lr_pretrained=lr_decay_factor * lr,  # decay factor scales base lr
                lr_new=lr,
                lr_dprnn=lr_dprnn,
                weight_decay=weight_decay
            )
        else:
            return self.parameter_groups_without_iter(
                lr_pretrained=lr_decay_factor * lr,  # decay factor scales base lr
                lr_new=lr,
                lr_dprnn=lr_dprnn,
                weight_decay=weight_decay
            )
