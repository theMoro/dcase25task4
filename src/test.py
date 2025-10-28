import os
import socket
import torch
import pathlib
import argparse

from munch import DefaultMunch
from pytorch_lightning.loggers import WandbLogger
import lightning.pytorch as pl

from src.utils import parse_yaml, initialize_config, ignore_warnings, setup_local_data_copy, update_dataset_paths_in_config
from src.paths import LOG_PATH, CKPT_PATH, DATA_PATH, LOCAL_DATA_PATH, NODE_LOCK_DIR_PATH

ignore_warnings()

def get_wandb_logger(config_dict, name=None, project="DCASE25_Task4", rank0_only=True, tags=None, entity=None):
    """Get wandb logger for logging test metrics."""
    if tags is None:
        tags = []
    if project is None:
        project = "DCASE25_Task4"

    config_filename = pathlib.Path(config_dict.get('config_yaml_file', 'unknown')).stem
    log_dir = os.path.join(LOG_PATH, config_filename)
    os.makedirs(log_dir, exist_ok=True)

    wandb_logger = WandbLogger(
        entity=entity,
        project=project,
        tags=tags,
        config=config_dict,
        name=name,
        dir=log_dir,
        log_model=False
    )

    return wandb_logger


def main(config_file, checkpoint_path, wandb_project="DCASE25_Task4", wandb_name=None):
    """Main testing function."""
    
    # Parse config
    config_dict = parse_yaml(config_file)
    config_dict['config_yaml_file'] = config_file
    
    # Determine rank for multi-GPU testing
    rank = 0
    if torch.distributed.is_initialized():
        rank = torch.distributed.get_rank()
    
    print("rank:", rank)
    
    # Setup local rank and nodes
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    print("local_rank:", local_rank)
    
    num_nodes = int(os.environ.get("SLURM_NNODES", 1))
    print("num_nodes:", num_nodes)
    if num_nodes > 1:
        print("Updating number of nodes:", num_nodes)
        config_dict['train']['trainer']['args']['num_nodes'] = num_nodes
    
    hostname = socket.gethostname()
    print("hostname:", hostname)
    
    # Setup data copying (can be disabled via config)
    enable_data_copy = config_dict.get('enable_data_copy', True)
    data_path_to_use = setup_local_data_copy(
        DATA_PATH, 
        LOCAL_DATA_PATH, 
        NODE_LOCK_DIR_PATH, 
        enable_copy=enable_data_copy
    )
    
    # Update dataset paths in config to use the appropriate data path
    update_dataset_paths_in_config(config_dict, data_path_to_use)
    
    # Setup logger
    logger = None
    if rank == 0:
        wandb_entity = config_dict.get('wandb_entity', None)
        logger = get_wandb_logger(config_dict, name=wandb_name, project=wandb_project, entity=wandb_entity)
    
    print("Config main: ")
    print(config_dict)
    
    configs = DefaultMunch.fromDict(config_dict)
    
    # Deterministic mode
    if configs.get('deterministic', False):
        torch.use_deterministic_algorithms(True, warn_only=True)
        pl.seed_everything(configs.get('manual_seed', 0), workers=True)
        configs['train']['trainer']['args']['deterministic'] = True
    
    config_filename = pathlib.Path(configs['config_yaml_file']).stem
    
    ckpt_save_path = os.path.join(CKPT_PATH, config_filename)
    os.makedirs(ckpt_save_path, exist_ok=True)
    run_id = len([f for f in os.listdir(ckpt_save_path) if f.startswith('run_')])

    if rank == 0:
        wandb_id = logger.experiment.id
        os.makedirs(os.path.join(ckpt_save_path, f"{wandb_id}"), exist_ok=True)
    
    # Initialize data module
    print('Initialize data module')
    data_module = initialize_config(configs['datamodule'])
    
    # Initialize lightning module
    print('Initialize lightning module')
    pl_model = initialize_config(configs['lightning_module'])
    
    # Load checkpoint
    print(f'Loading checkpoint: {checkpoint_path}')
    
    # For separator models, we need to preserve evaluation_tagger_model config
    # Check if this is a separator model (has evaluation_tagger_model)
    is_separator_model = hasattr(pl_model, 'evaluation_tagger_model') and pl_model.evaluation_tagger_model is not None
    
    if is_separator_model:
        # Store essential config attributes before loading checkpoint
        evaluation_tagger_model_config = pl_model.evaluation_tagger_model
        label_set = pl_model.label_set
        labels = pl_model.labels
        onehots = pl_model.onehots
        label_onehots = pl_model.label_onehots
        
        # Load checkpoint
        pl_model = pl_model.load_from_checkpoint(checkpoint_path)
        
        # Restore essential config attributes after loading checkpoint
        pl_model.evaluation_tagger_model = evaluation_tagger_model_config
        pl_model.eval_checkpoint = evaluation_tagger_model_config['args']['checkpoint']  # Re-derive from config
        pl_model.label_set = label_set
        pl_model.labels = labels
        pl_model.onehots = onehots
        pl_model.label_onehots = label_onehots
        
        print("Restored evaluation configuration after checkpoint loading")
    else:
        # For tagger models, just load the checkpoint normally
        pl_model = pl_model.load_from_checkpoint(checkpoint_path)
    
    # Initialize trainer
    configs['train']['trainer']['args']['logger'] = logger
    trainer = initialize_config(configs['train']['trainer'])
    
    print("Run ID: ", run_id)
    
    if rank == 0:
        logger.experiment.config.update({'run_id': run_id})
    
    # Test the model
    trainer.test(
        model=pl_model,
        datamodule=data_module
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test DCASE 2025 Task 4 models")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to checkpoint to test")
    parser.add_argument("--wandb_project", type=str, default="DCASE25_Task4", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    main(
        config_file=args.config,
        checkpoint_path=args.checkpoint,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )