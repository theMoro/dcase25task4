import os
import socket
import torch
import pathlib
import argparse

from munch import DefaultMunch
from pytorch_lightning.loggers import WandbLogger

from src.training.lightningmodule.helper import load_ckpt
from src.utils import parse_yaml, initialize_config, ignore_warnings, setup_local_data_copy, update_dataset_paths_in_config
from src.paths import DATA_PATH, LOCAL_DATA_PATH

# Import Lightning and other necessary modules
import lightning.pytorch as pl

ignore_warnings()

# Global variables for paths
LOG_PATH = os.path.join("outputs", "logs")
CKPT_PATH = os.path.join("outputs", "checkpoints")
NODE_LOCK_DIR_PATH = "/tmp/"

def get_wandb_logger(config_dict, name=None, project="DCASE25_Task4", rank0_only=True, tags=None, resume_wandb_id=None, entity=None):
    """Get wandb logger for logging training metrics."""
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
        id=resume_wandb_id,
        dir=log_dir,
        resume="must" if resume_wandb_id else None,
        log_model=False  # Don't store models on wandb
    )

    return wandb_logger


def main(config_file, resume_ckpt_path=None, wandb_project="DCASE25_Task4", wandb_name=None, resume_wandb_id=None,
         # Model configuration overrides
         max_iterations=None, use_time_film=None, use_dprnn=None, 
         embedding_injection_apply=None, embedding_injection_wandb_id=None, embedding_injection_checkpoint_path=None,
         embedding_injection_trainable=None, lr_dprnn=None,
         # Training configuration overrides
         train_batch_size=None, val_batch_size=None, learning_rate=None):
    """Main training function."""
    
    # Parse config
    config_dict = parse_yaml(config_file)
    config_dict['config_yaml_file'] = config_file

    if embedding_injection_wandb_id is not None and embedding_injection_checkpoint_path is not None:
        raise ValueError("Unclear which model to use, if both embedding_injection_wandb_id and embedding_injection_checkpoint_path are used!")

    # overwrite wandb_id used for embedding_injection
    if embedding_injection_wandb_id is not None:
        ckpt_in_yaml_file = config_dict["lightning_module"]["args"]["embedding_injection"]["tagger_model"]["args"]["checkpoint"]

        if ckpt_in_yaml_file != "strong":
            raise Warning(f"Overwriting checkpoint defined in YAML file ({ckpt_in_yaml_file}) with embedding_injection_wandb_id: {embedding_injection_wandb_id}!")

        config_dict["lightning_module"]["args"]["embedding_injection"]["tagger_model"]["args"]["checkpoint"] = embedding_injection_wandb_id

    # Determine rank for multi-GPU training
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
    
    # Apply command-line overrides to config
    if max_iterations is not None:
        config_dict['lightning_module']['args']['model']['args']['max_iterations'] = max_iterations
    if use_time_film is not None:
        config_dict['lightning_module']['args']['model']['args']['use_time_film'] = use_time_film
    if use_dprnn is not None:
        config_dict['lightning_module']['args']['model']['args']['dprnn']['use'] = use_dprnn
        if use_dprnn and lr_dprnn is not None:
            config_dict['lightning_module']['args']['optimizer']['lr_dprnn'] = lr_dprnn
        elif not use_dprnn:
            config_dict['lightning_module']['args']['optimizer']['lr_dprnn'] = None
    if embedding_injection_apply is not None:
        config_dict['lightning_module']['args']['embedding_injection']['apply'] = embedding_injection_apply
    if embedding_injection_trainable is not None:
        config_dict['lightning_module']['args']['embedding_injection']['trainable'] = embedding_injection_trainable
    if train_batch_size is not None:
        config_dict['datamodule']['args']['train_dataloader']['batch_size'] = train_batch_size
    if val_batch_size is not None:
        config_dict['datamodule']['args']['val_dataloader']['batch_size'] = val_batch_size
    if learning_rate is not None:
        config_dict['lightning_module']['args']['optimizer']['args']['lr'] = learning_rate
    
    # Setup logger
    logger = None
    if rank == 0:
        wandb_entity = config_dict.get('wandb_entity', None)
        logger = get_wandb_logger(config_dict, name=wandb_name, project=wandb_project, resume_wandb_id=resume_wandb_id, entity=wandb_entity)
    
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
    
    wandb_id = None
    if rank == 0:
        wandb_id = logger.experiment.id
        os.makedirs(os.path.join(ckpt_save_path, f"{wandb_id}"), exist_ok=True)
    
    # Initialize data module
    print('Initialize data module')
    data_module = initialize_config(configs['datamodule'])
    
    # Initialize lightning module
    print('Initialize lightning module')
    pl_model = initialize_config(configs['lightning_module'])

    if embedding_injection_checkpoint_path is not None:
        load_ckpt(embedding_injection_checkpoint_path, pl_model.embedding_model)
    
    # Initialize callbacks
    print('Initialize callbacks')
    callbacks_configs = configs['train']['callbacks']
    callbacks = []
    for callback_config in callbacks_configs:
        if callback_config['name'] == 'checkpoint':
            callback_config['args']['dirpath'] = os.path.join(ckpt_save_path, f"{wandb_id}") if wandb_id else os.path.join(ckpt_save_path, f"run_{run_id}")
        callback = initialize_config(callback_config)
        callbacks.append(callback)
    
    # Initialize trainer
    configs['train']['trainer']['args']['callbacks'] = callbacks
    configs['train']['trainer']['args']['logger'] = logger
    trainer = initialize_config(configs['train']['trainer'])
    
    print("Run ID: ", run_id)
    
    if rank == 0:
        logger.experiment.config.update({'run_id': run_id})
    
    # Setup resume checkpoint path
    if resume_ckpt_path:
        print(f'Resuming run from checkpoint: {resume_ckpt_path}')
        ckpt_resume_path = resume_ckpt_path
    else:
        ckpt_resume_path = None

    if configs['deterministic']: torch.use_deterministic_algorithms(True, warn_only=True)
    
    # Enable torch flags for optimization
    if configs['train']['trainer'].get('torch_flags', False):
        print("⚡ Mixed precision enabled — enabling all attention backends")
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        torch.backends.cuda.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    
    # Channels last memory format
    if configs['train']['trainer'].get('channels_last', False):
        pl_model = pl_model.to(memory_format=torch.channels_last)
    
    # Compile model
    if configs['train']['trainer'].get('compile', False):
        try:
            print("🔧 Compiling model with torch.compile...")
            pl_model = torch.compile(pl_model, mode="default", fullgraph=False)
        except Exception as e:
            print(f"⚠️ torch.compile failed: {e}")
    
    # Train the model
    trainer.fit(
        model=pl_model,
        train_dataloaders=None,
        val_dataloaders=None,
        datamodule=data_module,
        ckpt_path=ckpt_resume_path
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train DCASE 2025 Task 4 models")
    parser.add_argument("--config", "-c", type=str, required=True, help="Path to config YAML file")
    parser.add_argument("--resume", "-r", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--wandb_project", type=str, default="DCASE25_Task4", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    parser.add_argument("--resume_wandb_id", type=str, default=None, help="Wandb ID to resume from")
    
    # Model configuration overrides
    parser.add_argument("--max_iterations", type=int, default=None, help="Maximum iterations for iterative models")
    parser.add_argument("--use_time_film", action="store_true", default=None, help="Enable TimeFiLM")
    parser.add_argument("--no_time_film", action="store_true", help="Disable TimeFiLM")
    parser.add_argument("--use_dprnn", action="store_true", default=None, help="Enable DPRNN")
    parser.add_argument("--no_dprnn", action="store_true", help="Disable DPRNN")
    parser.add_argument("--embedding_injection_apply", action="store_true", default=None, help="Enable Embedding Injection")
    parser.add_argument("--embedding_injection_wandb_id", default=None, type=str, help="Wandb ID of model used for Embedding Injection")
    parser.add_argument("--embedding_injection_checkpoint_path", default=None, type=str, help="Path to checkpoint of model used for Embedding Injection")
    parser.add_argument("--no_embedding_injection", action="store_true", help="Disable Embedding Injection")
    parser.add_argument("--embedding_injection_trainable", action="store_true", default=None, help="Make Embedding Injection trainable")
    parser.add_argument("--embedding_injection_frozen", action="store_true", help="Freeze Embedding Injection")
    parser.add_argument("--lr_dprnn", type=float, default=None, help="Learning rate for DPRNN")
    
    # Training configuration overrides
    parser.add_argument("--train_batch_size", type=int, default=None, help="Training batch size")
    parser.add_argument("--val_batch_size", type=int, default=None, help="Validation batch size")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate")
    
    args = parser.parse_args()
    
    # Handle boolean overrides
    if args.no_time_film:
        args.use_time_film = False
    if args.no_dprnn:
        args.use_dprnn = False
    if args.no_embedding_injection:
        args.embedding_injection_apply = False
    if args.embedding_injection_frozen:
        args.embedding_injection_trainable = False
    
    main(
        config_file=args.config,
        resume_ckpt_path=args.resume,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        resume_wandb_id=args.resume_wandb_id,
        max_iterations=args.max_iterations,
        use_time_film=args.use_time_film,
        use_dprnn=args.use_dprnn,
        embedding_injection_apply=args.embedding_injection_apply,
        embedding_injection_wandb_id=args.embedding_injection_wandb_id,
        embedding_injection_checkpoint_path=args.embedding_injection_checkpoint_path,
        embedding_injection_trainable=args.embedding_injection_trainable,
        lr_dprnn=args.lr_dprnn,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        learning_rate=args.learning_rate
    )