from src.models.pretrained_sed_models.m2d.portable_m2d import PortableM2D as M2D
from src.models.transformer_wrapper import BaseModelWrapper
import torch


class M2DWrapper(BaseModelWrapper):
    def __init__(self) -> None:
        super().__init__()
        self.m2d = M2D()

    def mel_forward(self, x):
        return self.m2d.to_normalized_feature(x)

    def forward(self, spec, return_features=False):
        return self.m2d.forward_mel(spec, return_features=return_features)

    def _get_top_prob(self, prob_vector, pthre, nevent_range):
        # prob_vector [bs, nclasses]
        batch_size, n_classes = prob_vector.shape
        onehots = torch.zeros_like(prob_vector)

        # Get the top probabilities and their indices
        for i in range(batch_size):
            probs = prob_vector[i]
            selected_indices = (probs >= pthre).nonzero(as_tuple=True)[0]

            if len(selected_indices) < nevent_range[0]:  # Select at least nevent_range[0] with highest probability
                top_indices = torch.argsort(probs, descending=True)[:nevent_range[0]]
            else:  # Select at most nevent_range[1] from those exceeding threshold
                top_indices = selected_indices[
                    torch.argsort(probs[selected_indices], descending=True)[:nevent_range[1]]]
            onehots[i, top_indices] = 1.
        return onehots

    def predict(self, input_dict, pthre=0.5, nevent_range=[1, 3]):  # remove the above function later
        if input_dict['waveform'].dim() == 3:  # [bs, nch, wlen]
            input_dict['waveform'] = input_dict['waveform'][:, 0, :]
        output_dict = {}
        output_dict['probabilities'] = self.forward(input_dict["waveform"])
        output_dict['multihot_vector'] = self._get_top_prob(output_dict['probabilities'], pthre, nevent_range)
        return output_dict

    def separate_params(self):
        pt_params = [[], [], [], [], [], [], [], [], [], [], [], []]
        for k, p in self.named_parameters():
            if any(['cls_token' in k,
                    'pos_embed' in k,
                    'norm_stats' in k,
                    'patch_embed' in k]):
                pt_params[0].append(p)
            elif 'blocks.0.' in k:
                pt_params[0].append(p)
            elif 'blocks.1.' in k:
                pt_params[1].append(p)
            elif 'blocks.2.' in k:
                pt_params[2].append(p)
            elif 'blocks.3.' in k:
                pt_params[3].append(p)
            elif 'blocks.4.' in k:
                pt_params[4].append(p)
            elif 'blocks.5.' in k:
                pt_params[5].append(p)
            elif 'blocks.6.' in k:
                pt_params[6].append(p)
            elif 'blocks.7.' in k:
                pt_params[7].append(p)
            elif 'blocks.8.' in k:
                pt_params[8].append(p)
            elif 'blocks.9.' in k:
                pt_params[9].append(p)
            elif 'blocks.10.' in k:
                pt_params[10].append(p)
            elif 'blocks.11.' in k:
                pt_params[11].append(p)
            elif 'backbone.norm.weight' in k or 'backbone.norm.bias' in k:
                pt_params[11].append(p)
            else:
                raise ValueError(f"Check separate params for M2D! Unknown key: {k}")
        return list(reversed(pt_params))
