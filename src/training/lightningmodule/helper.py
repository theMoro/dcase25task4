import os

import torch
import wandb


def _map_soundbeam_to_embedding_injection(state_dict):
    """
    Map old 'soundbeam' parameter names to new 'embedding_injection' names for backward compatibility.
    
    Args:
        state_dict: Dictionary of parameter names to tensors
        
    Returns:
        Dictionary with updated parameter names
    """
    mapped_dict = {}
    for key, value in state_dict.items():
        # Replace 'soundbeam' with 'embedding_injection' in parameter names
        new_key = key.replace('soundbeam_weights', 'embedding_injection_weights')
        new_key = new_key.replace('weight_soundbeam_features', 'weight_embedding_injection_features')
        new_key = new_key.replace('soundbeam_merge_method', 'embedding_injection_merge_method')
        # General replacement for any remaining 'soundbeam' references
        new_key = new_key.replace('soundbeam', 'embedding_injection')
        mapped_dict[new_key] = value
    return mapped_dict


def load_ckpt(path, model):
    model_ckpt = torch.load(path, weights_only=False, map_location='cpu')
    model_ckpt = model_ckpt['state_dict']

    # Map old 'soundbeam' parameter names to 'embedding_injection' for backward compatibility
    model_ckpt = _map_soundbeam_to_embedding_injection(model_ckpt)

    if set(model.state_dict().keys()) != set \
            (model_ckpt.keys()): # remove prefix, incase the ckpt is of lightning module
        one_model_key = next(iter(model.state_dict().keys()))

        ckpt_corresponding_key = [k for k in model_ckpt.keys() if k.endswith(one_model_key)]

        prefix = ckpt_corresponding_key[0][:-len(one_model_key)]
        model_ckpt = {k[len(prefix):]: v for k, v in model_ckpt.items() if k.startswith(prefix)  }# remove prefix

    model.load_state_dict(model_ckpt)


def load_ckpt_wandb(checkpoint_id, model):
    artifact_dir = f"artifacts/{checkpoint_id}:v0"
    artifact_spec = f"cp_tobi/DCASE25_Task4/{checkpoint_id}:v0"

    found_path = None

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

    load_ckpt(found_path, model)