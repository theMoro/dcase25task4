import os
import socket

hostname = socket.gethostname()

RESOURCES_FOLDER = "resources"
CHECKPOINT_URL = "https://github.com/fschmid56/PretrainedSED/releases/download/v0.0.1/"

DATA_PATH = '/path/to/dataset'  # TODO: replace with actual dataset path
LOCAL_DATA_PATH = DATA_PATH

LOG_PATH = os.path.join("outputs", "logs")
CKPT_PATH = os.path.join("outputs", "checkpoints")
NODE_LOCK_DIR_PATH = "/tmp/"