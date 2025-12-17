import os
import socket

hostname = socket.gethostname()

RESOURCES_FOLDER = "resources"
CHECKPOINT_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"

if hostname in ['ln01'] or hostname.startswith('node'):  # LIT
    DATA_PATH = '/system/user/clusterscratch/dcase25_task4_v2/DCASE2025Task4Dataset/dev_set/'
    LOCAL_DATA_PATH = '/tmp/local_data_dcase/dev_set/'

    LOG_PATH = os.path.join("outputs", "logs")
    CKPT_PATH = os.path.join("outputs", "checkpoints")
    NODE_LOCK_DIR_PATH = "/tmp/"
elif "vsc" in hostname or hostname.startswith("n"):
    DATA_PATH = '/gpfs/data/fs72765/tobiasm/dcase25_task4/DCASE2025Task4Dataset/dev_set/'  # $DATA
    LOCAL_DATA_PATH = '/tmp/local_dcase_data/dev_set/'

    LOG_PATH = os.path.join("outputs", "logs")
    CKPT_PATH = '/gpfs/data/fs72783/paul_s5/checkpoints'
    NODE_LOCK_DIR_PATH = "/tmp/"
elif "rk" in hostname or "ed" in hostname: # RKs
    DATA_PATH = '/opt/scratch/dcase25_task4/data/dev_set/'
    EVAL_DATA_PATH = '/opt/scratch/dcase25_task4/data/eval_set/'

    # username = os.getlogin()
    import getpass
    username = getpass.getuser()
    LOCAL_DATA_PATH = f'/var/home/{username}/dcase25/local_dcase_data/dev_set/'

    LOG_PATH = f"/var/home/{username}/dcase25/logs"
    CKPT_PATH = f"/var/home/{username}/dcase25/checkpoints"
    FINAL_CKPT_PATH = f"/opt/scratch/dcase25_task4/final_checkpoints"
    PREDICTIONS_CKPT_PATH = f"/opt/scratch/dcase25_task4/predictions"
    NODE_LOCK_DIR_PATH = "/tmp/"
elif hostname == "k3pc":
    DATA_PATH = '/share/opt/scratch/dcase25_task4/data/dev_set/'
    LOCAL_DATA_PATH = DATA_PATH

    LOG_PATH = os.path.join("outputs", "logs")
    CKPT_PATH = os.path.join("outputs", "checkpoints")
    NODE_LOCK_DIR_PATH = "/tmp/"
else:
    DATA_PATH = '/share/opt/scratch/dcase25_task4/data/dev_set/'
    LOCAL_DATA_PATH = DATA_PATH

    LOG_PATH = os.path.join("outputs", "logs")
    CKPT_PATH = os.path.join("outputs", "checkpoints")
    NODE_LOCK_DIR_PATH = "/tmp/"

# DATA_PATH = '/path/to/dataset'  # TODO: replace with actual dataset path
# LOCAL_DATA_PATH = DATA_PATH

LOG_PATH = os.path.join("outputs", "logs")
CKPT_PATH = os.path.join("outputs", "checkpoints")
NODE_LOCK_DIR_PATH = "/tmp/"