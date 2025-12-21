import os
import socket
import torch
import pathlib
import argparse

from munch import DefaultMunch
from pytorch_lightning.loggers import WandbLogger
import lightning.pytorch as pl

from src.training.lightningmodule.helper import load_ckpt, load_ckpt_wandb
from src.utils import parse_yaml, initialize_config, ignore_warnings, setup_local_data_copy
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


def main(config_file, tagger_wandb_id=None, tagger_checkpoint_path=None, separator_checkpoint_path=None, sep_multiple_iter=None, max_iterations=1, wandb_project="DCASE25_Task4", wandb_name=None):
    """Main testing function."""

    if tagger_wandb_id is not None and tagger_checkpoint_path is not None:
        raise ValueError("Unclear which tagger model to use if both tagger_wandb_id and tagger_checkpoint_path are specified!")

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
    config_test = config_dict['datamodule']['args']['test_dataloader']['dataset']['args']['soundscape_dir']
    config_dict['datamodule']['args']['test_dataloader']['dataset']['args']['soundscape_dir'] = os.path.join(
        data_path_to_use, config_test[len('data/dev_set/'):])

    config_test = config_dict['datamodule']['args']['test_dataloader']['dataset']['args']['oracle_target_dir']
    config_dict['datamodule']['args']['test_dataloader']['dataset']['args']['oracle_target_dir'] = os.path.join(
        data_path_to_use, config_test[len('data/dev_set/'):])
    
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

    if sep_multiple_iter:
        configs["lightning_module"]["args"]["model"]["module"] = "src.models.audiosep_resunet_iter.resunet"
        configs["lightning_module"]["args"]["model"]["args"]["max_iterations"] = max_iterations
        configs["lightning_module"]["args"]["model"]["args"]["dprnn"]["use"] = True  # because the iterative models were trained with DPRNN

    pl_model = initialize_config(configs['lightning_module'])

    # For separator models, we need to preserve evaluation_tagger_model config
    # Check if this is a separator model (has evaluation_tagger_model)
    evaluate_separator = hasattr(pl_model, 'evaluation_tagger_model') and pl_model.evaluation_tagger_model is not None

    if evaluate_separator:
        print(f"Loading checkpoint of separator model and corresponding embedding model: {separator_checkpoint_path}")
        load_ckpt(separator_checkpoint_path, pl_model.model)
        load_ckpt(separator_checkpoint_path, pl_model.embedding_model)  # the weights of the embedding model are in the ckpt of the separation model

        if configs['lightning_module']['args']['evaluation_tagger_model']['args']['checkpoint'] == 'baseline':
            print("Loading baseline checkpoint for baseline M2D tagger model")
            tagger_ckpt = "checkpoint/m2dat.ckpt"
            load_ckpt(tagger_ckpt, pl_model.evaluation_tagger_model)
        else:
            if tagger_wandb_id:
                print(f"Loading checkpoint {tagger_wandb_id} for our M2D tagger model")
                load_ckpt_wandb(tagger_wandb_id, pl_model.evaluation_tagger_model)
            elif tagger_checkpoint_path:
                print(f"Loading following checkpoint for our M2D tagger model: {tagger_checkpoint_path}")
                load_ckpt(tagger_checkpoint_path, pl_model.evaluation_tagger_model)
            else:
                raise Exception("tagger_wandb_id should not be None in this case!")
    else:
        if tagger_wandb_id:
            print(f"Loading checkpoint {tagger_wandb_id} for our M2D tagger model")
            load_ckpt_wandb(tagger_wandb_id, pl_model.model)
        elif tagger_checkpoint_path:
            print(f"Loading following checkpoint for our M2D tagger model: {tagger_checkpoint_path}")
            load_ckpt(tagger_checkpoint_path, pl_model.model)
        else:
            raise Exception("tagger_wandb_id should not be None in this case!")
    
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
    parser.add_argument("--tagger_wandb_id", type=str, required=False, default=None, help="wandb_id of tagger model to test")
    parser.add_argument("--tagger_checkpoint_path", type=str, required=False, default=None, help="Path to checkpoint of tagger model to test")
    parser.add_argument("--separator_checkpoint_path", type=str, required=False, default=None, help="Path to checkpoint of separator model to test")
    parser.add_argument("--sep_multiple_iter", action="store_true", default=None, help="Use separation model where multiple iterations are possible")
    parser.add_argument("--max_iterations", type=int, default=1, help="Number of iterations used for inference for the separation model")
    parser.add_argument("--wandb_project", type=str, default="DCASE25_Task4", help="Wandb project name")
    parser.add_argument("--wandb_name", type=str, default=None, help="Wandb run name")
    
    args = parser.parse_args()
    
    main(
        config_file=args.config,
        tagger_wandb_id=args.tagger_wandb_id,
        tagger_checkpoint_path=args.tagger_checkpoint_path,
        separator_checkpoint_path=args.separator_checkpoint_path,
        sep_multiple_iter=args.sep_multiple_iter,
        max_iterations=args.max_iterations,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name
    )