import os
import sys
import datetime
import logging
from typing import Dict, List, NoReturn
import yaml
import importlib
import pytz
import torch
import shutil
import socket
import stat
from filelock import FileLock

LABELS = {
    'dcase2025t4': ["AlarmClock", "BicycleBell", "Blender", "Buzzer",
                    "Clapping", "Cough", "CupboardOpenClose", "Dishes",
                    "Doorbell", "FootSteps", "HairDryer", "MechanicalFans",
                    "MusicalKeyboard", "Percussion", "Pour", "Speech",
                    "Typing", "VacuumCleaner"],
}


def ignore_warnings():
    import warnings

    # Filter out warnings containing "deprecated"
    warnings.filterwarnings("ignore", message=".*invalid value encountered in cast.*")
    warnings.filterwarnings("ignore", message=".*deprecated.*")
    warnings.filterwarnings("ignore", message=".*torch.meshgrid.*")
    warnings.filterwarnings("ignore", message=".*that has Tensor Cores.*")
    warnings.filterwarnings("ignore", message=".*cudnnFinalize Descriptor Failed.*")
    warnings.filterwarnings("ignore", message=".*torch.use_deterministic_algorithms(True, warn_only=True)*")
    from transformers import logging
    logging.set_verbosity_error()


logging_levels = {
    "CRITICAL": logging.CRITICAL,
    "ERROR": logging.ERROR,
    "WARNING": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "NOTSET": logging.NOTSET
}


def logging_setup(directory, file_log_level="INFO", console_log_level='DEBUG', timezone='Europe/Vienna'):
    file_log_level = logging_levels[file_log_level]
    console_log_level = logging_levels[console_log_level]
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    # create directory, generate logfilename
    os.makedirs(directory, exist_ok=True)
    a = datetime.datetime.now().astimezone(pytz.timezone(timezone))
    logfilename = '%04d%02d%02d_%02dh%02d.log' % (a.year, a.month, a.day, a.hour, a.minute)
    log_full_path = os.path.join(directory, logfilename)

    # setup log to file
    file_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s: %(message)s')
    file_handler = logging.FileHandler(log_full_path)
    file_handler.setLevel(file_log_level)
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # print to console
    console_formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(name)s (%(processName)s): %(message)s')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(console_log_level)  # always debug level
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)


def parse_yaml(config_yaml) -> Dict:
    r"""Parse yaml file.

    Args:
        config_yaml (str or None): config yaml path

    Returns:
        yaml_dict (Dict): parsed yaml file
    """

    with open(config_yaml, "r") as fr:
        return yaml.load(fr, Loader=yaml.FullLoader)


def initialize_config(module_cfg, reload=False):
    if reload and module_cfg["module"] in sys.modules:
        module = importlib.reload(sys.modules[module_cfg["module"]])
    else:
        module = importlib.import_module(module_cfg["module"])
    if 'args' in module_cfg.keys(): return getattr(module, module_cfg["main"])(**module_cfg["args"])
    return getattr(module, module_cfg["main"])()


def lightning_load_from_checkpoint(lightning_module_cfg, ckpt_path):
    module = importlib.import_module(lightning_module_cfg["module"])
    model_class = getattr(module, lightning_module_cfg["main"])
    model = model_class.load_from_checkpoint(ckpt_path, **lightning_module_cfg['args'])
    return model


def ca_metric(est_lb, est_wf, ref_lb, ref_wf, mixture, metricfunc):
    """
    Permutation-sensitive metric calculation.

    Args:
        est_lb (list of str): Labels for estimated sources, length n_est_sources.
        est_wf (torch.Tensor): Corresponding estimated waveforms, shape [n_est_sources, wf_len].
        ref_lb (list of str): Labels for reference sources, length n_ref_sources.
        ref_wf (torch.Tensor): Corresponding reference waveforms, shape [n_ref_sources, wf_len].
        mixture (torch.Tensor): Mixture waveform at the reference channel, shape [wf_len].
        metricfunc (callable): metric function, metricfunc(predicted, target)
    """

    est_dict = {lb: wf for lb, wf in zip(est_lb, est_wf)}
    ref_dict = {lb: wf for lb, wf in zip(ref_lb, ref_wf)}

    tp_lb = set(est_lb) & set(ref_lb) - {'silence'}  # true positive labels
    fp_lb = set(est_lb) - tp_lb - {'silence'}  # false positive labels
    fn_lb = set(ref_lb) - tp_lb - {'silence'}  # false negative labels

    result = torch.tensor([])
    if tp_lb:
        tp_est_wf = torch.stack([est_dict[lb] for lb in tp_lb])  # shape [len(tp_lb), wf_len]
        tp_ref_wf = torch.stack([ref_dict[lb] for lb in tp_lb])  # shape [len(tp_lb), wf_len]
        repeated_mixture = mixture.unsqueeze(0).repeat(len(tp_lb), 1)
        score_est = metricfunc(tp_est_wf, tp_ref_wf)
        score_mixture = metricfunc(repeated_mixture, tp_ref_wf)
        improvement = score_est - score_mixture  # shape [len(tp_lb)]
        result = torch.cat((result, improvement), dim=0)
    pens = torch.zeros(len(fp_lb) + len(fn_lb))
    result = torch.cat((result, pens), dim=0)

    return result.mean().item()


def setup_local_data_copy(data_path, local_data_path, node_lock_dir_path, enable_copy=True):
    """
    Copy dataset to local storage for faster training/evaluation.
    
    This function implements a thread-safe mechanism to copy the dataset from network storage
    to local storage. Training is significantly faster when using local storage because:
    - Reduced I/O latency (local SSD vs network storage)
    - Higher bandwidth (local storage vs network)
    - Reduced network congestion in multi-node setups
    
    Args:
        data_path (str): Path to the original dataset on network storage
        local_data_path (str): Path to copy dataset to on local storage
        node_lock_dir_path (str): Directory for lock files
        enable_copy (bool): Whether to enable data copying (default: True)
    
    Returns:
        str: The path to use for dataset access (local_data_path if copying enabled, data_path otherwise)
    """
    if not enable_copy:
        print("Data copying disabled, using original data path:", data_path)
        return data_path
    
    hostname = socket.gethostname()
    
    # Get username for lock file
    try:
        import getpass
        username = getpass.getuser()
    except Exception as e:
        print(f"Error getting username: {e}")
        username = "unknown_user"
    
    # Setup local rank for multi-GPU processing
    local_rank = int(os.environ.get("SLURM_LOCALID", 0))
    
    lock_file_path = os.path.join(node_lock_dir_path, f'copy_data_{hostname}_{username}.lock')
    lock = FileLock(lock_file_path)
    
    # Acquire lock and copy data to local path if needed
    with lock:
        os.chmod(lock_file_path, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | 
                 stat.S_IWGRP | stat.S_IROTH | stat.S_IWOTH)
        
        print(f"Local Rank: {local_rank} acquired copy lock.")
        
        if not os.path.exists(local_data_path):
            print("Copying data to local path...", flush=True)
            shutil.copytree(data_path, local_data_path, copy_function=shutil.copy)
            print("Finished copying data to local path!", flush=True)
        
        print(f"Local Rank: {local_rank} released copy lock.")
    
    print("DATASET PATH: ", local_data_path, flush=True)
    return local_data_path


def update_dataset_paths_in_config(config_dict, local_data_path):
    """
    Update dataset paths in config to use local storage.
    
    Args:
        config_dict (dict): Configuration dictionary to update
        local_data_path (str): Local data path to use for dataset paths
    """
    # Update training dataset paths
    train_config = config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['config']
    
    # Update foreground directory
    foreground_dir = train_config['spatialscaper']['foreground_dir']
    if config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_full']:
        foreground_dir = '/'.join(foreground_dir.split('/')[:-1] + ['full'])
    train_config['spatialscaper']['foreground_dir'] = os.path.join(
        local_data_path, foreground_dir[len('data/dev_set/'):])
    
    # Update background directory
    background_dir = train_config['spatialscaper']['background_dir']
    if config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_full']:
        if config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_additional_backgrounds']:
            background_dir = '/'.join(background_dir.split('/')[:-1] + ['full_plus'])
        else:
            background_dir = '/'.join(background_dir.split('/')[:-1] + ['full'])
    train_config['spatialscaper']['background_dir'] = os.path.join(
        local_data_path, background_dir[len('data/dev_set/'):])
    
    # Update RIR directory
    rir_dir = train_config['spatialscaper']['rir_dir']
    if config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_full']:
        ir_folder = config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_additional_irs']
        if ir_folder:
            assert ir_folder in ['full_plus_t', 'full_plus_ts'], "use_additional_irs must be 'full_t' or 'full_ts'"
            rir_dir = '/'.join(rir_dir.split('/')[:-1] + [ir_folder])
        else:
            rir_dir = '/'.join(rir_dir.split('/')[:-1] + ['full'])
    train_config['spatialscaper']['rir_dir'] = os.path.join(
        local_data_path, rir_dir[len('data/dev_set/'):])
    
    # Update interference directory
    interference_dir = train_config['spatialscaper']['interference_dir']
    if config_dict['datamodule']['args']['train_dataloader']['dataset']['args']['use_full']:
        interference_dir = '/'.join(interference_dir.split('/')[:-1] + ['full'])
    train_config['spatialscaper']['interference_dir'] = os.path.join(
        local_data_path, interference_dir[len('data/dev_set/'):])
    
    # Update validation dataset path
    config_val = config_dict['datamodule']['args']['val_dataloader']['dataset']['args']['config']
    config_dict['datamodule']['args']['val_dataloader']['dataset']['args']['config'] = os.path.join(
        local_data_path, config_val[len('data/dev_set/'):])
