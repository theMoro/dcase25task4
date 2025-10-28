import torch

def load_ckpt(path, model):
    model_ckpt = torch.load(path, weights_only=False, map_location='cpu')
    model_ckpt = model_ckpt['state_dict']
    if set(model.state_dict().keys()) != set \
            (model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
        one_model_key = next(iter(model.state_dict().keys()))
        ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]
        prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
        model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)  }# remove prefix
    model.load_state_dict(model_ckpt)